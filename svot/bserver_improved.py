import cv2
import numpy as np
import socket
import struct
import threading
import time

# ================= FEATURE MATCHER =================
class FeatureMatcher:
    """Fast feature matching for alignment verification."""
    
    def __init__(self, detector_type='ORB'):
        """
        Args:
            detector_type: 'ORB' (fastest), 'AKAZE' (good balance), or 'SIFT' (best quality)
        """
        self.detector_type = detector_type
        
        if detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1000, fastThreshold=10)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # SIFT
            self.detector = cv2.SIFT_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def match_and_align(self, frame, map_region, min_matches=20):
        """
        Match features and compute homography refinement.
        
        Args:
            frame: New frame to align (BGR)
            map_region: Existing map region (BGR)
            min_matches: Minimum inliers required for valid alignment
        
        Returns:
            dict with keys: 'success', 'homography', 'inliers', 'inlier_ratio'
        """
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        map_gray = cv2.cvtColor(map_region, cv2.COLOR_BGR2GRAY)
        
        # Detect and compute features
        kp1, des1 = self.detector.detectAndCompute(frame_gray, None)
        kp2, des2 = self.detector.detectAndCompute(map_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return {'success': False, 'homography': None, 'inliers': 0, 'inlier_ratio': 0.0}
        
        # Match features with ratio test
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < min_matches:
            return {'success': False, 'homography': None, 'inliers': 0, 'inlier_ratio': 0.0}
        
        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return {'success': False, 'homography': None, 'inliers': 0, 'inlier_ratio': 0.0}
        
        # Count inliers
        inliers = int(mask.sum())
        inlier_ratio = inliers / len(good_matches)
        
        return {
            'success': inliers >= min_matches,
            'homography': H,
            'inliers': inliers,
            'inlier_ratio': inlier_ratio,
            'total_matches': len(good_matches)
        }

# ================= QUALITY ASSESSMENT =================
class FrameQualityAssessor:
    """Assess frame quality for intelligent blending."""
    
    @staticmethod
    def assess_sharpness(frame):
        """Measure image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 (typical range: 0-2000 for aerial images)
        return min(laplacian_var / 1000.0, 1.0)
    
    @staticmethod
    def assess_exposure(frame):
        """Check if image is properly exposed (not too dark/bright)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        # Penalize very dark (<50) or very bright (>200) images
        if mean_brightness < 50 or mean_brightness > 200:
            return 0.3
        # Optimal range: 80-180
        deviation = abs(mean_brightness - 130) / 130.0
        return max(0.3, 1.0 - deviation)
    
    @staticmethod
    def compute_quality_score(frame, altitude, viewing_angle_deg):
        """
        Compute overall quality score (0-1).
        
        Args:
            frame: Input image
            altitude: Camera altitude in meters (negative in NED)
            viewing_angle_deg: Angle from nadir in degrees
        """
        sharpness = FrameQualityAssessor.assess_sharpness(frame)
        exposure = FrameQualityAssessor.assess_exposure(frame)
        
        # Prefer images taken from optimal altitude (20-80m)
        abs_alt = abs(altitude)
        altitude_score = 1.0
        if abs_alt < 10:
            altitude_score = 0.3  # Too close
        elif abs_alt > 120:
            altitude_score = max(0.3, 1.0 - (abs_alt - 120) / 200.0)  # Too far
        
        # Prefer near-nadir images (within 20 degrees)
        angle_score = max(0.2, np.cos(np.radians(viewing_angle_deg)))
        
        # Weighted combination
        quality = (
            0.35 * sharpness +
            0.25 * exposure +
            0.20 * altitude_score +
            0.20 * angle_score
        )
        
        return quality

# ================= MAPPER ENGINE =================
class MultiBandMap2D:
    def __init__(self, resolution=0.05, band_num=2, tile_size=512, 
                 enable_alignment_check=True, min_quality=0.3):
        """
        Multi-band blending mapper for aerial imagery with alignment verification.
        
        Args:
            resolution: meters per pixel in output map
            band_num: number of pyramid levels for blending
            tile_size: size of each map tile in pixels
            enable_alignment_check: whether to verify alignment with feature matching
            min_quality: minimum quality score to accept a frame (0-1)
        """
        self.resolution = resolution
        self.band_num = band_num
        self.tile_size = tile_size
        self.tiles = {}
        self.weight_mask = None
        self.last_frame_shape = None
        self.lock = threading.Lock()
        self.paused = False
        
        # Feature matching for alignment verification
        self.enable_alignment_check = enable_alignment_check
        self.feature_matcher = FeatureMatcher(detector_type='ORB')  # ORB is fastest
        self.min_quality = min_quality
        
        # Statistics
        self.frames_processed = 0
        self.frames_rejected_alignment = 0
        self.frames_rejected_quality = 0

    def _get_weight_mask(self, shape):
        """Create Gaussian-like weight mask favoring image center."""
        if self.weight_mask is not None and self.last_frame_shape == shape:
            return self.weight_mask
        h, w = shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        maxd = np.sqrt(center_x**2 + center_y**2)
        mask = np.clip(1.0 - dist / maxd, 1e-5, 1.0).astype(np.float32)
        self.weight_mask = mask * mask  # Squared for stronger center bias
        self.last_frame_shape = shape
        return self.weight_mask

    def _create_laplace_pyr(self, img):
        """Create Laplacian pyramid for multi-band blending."""
        pyr, cur = [], img
        for _ in range(self.band_num):
            down = cv2.pyrDown(cur)
            up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            pyr.append(cv2.subtract(cur, up))
            cur = down
        pyr.append(cur)
        return pyr
    
    def _get_map_region(self, xmin, xmax, ymin, ymax, margin=20):
        """
        Extract a region from the existing map for alignment checking.
        
        Returns None if no map exists in that region yet.
        """
        with self.lock:
            if not self.tiles:
                return None
            
            def tile_index(v):
                return int(np.floor(v / (self.resolution * self.tile_size)))
            
            tminx, tmaxx = tile_index(xmin - margin), tile_index(xmax + margin)
            tminy, tmaxy = tile_index(ymin - margin), tile_index(ymax + margin)
            
            # Check if any tiles exist in this region
            relevant_tiles = [(tx, ty) for tx in range(tminx, tmaxx + 1)
                             for ty in range(tminy, tmaxy + 1)
                             if (tx, ty) in self.tiles]
            
            if not relevant_tiles:
                return None
            
            # Create temporary canvas for this region
            canvas_w = int((xmax - xmin + 2 * margin) / self.resolution)
            canvas_h = int((ymax - ymin + 2 * margin) / self.resolution)
            
            if canvas_w <= 0 or canvas_h <= 0 or canvas_w > 10000 or canvas_h > 10000:
                return None
            
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # Render relevant tiles
            for (tx, ty) in relevant_tiles:
                tile_x_world = tx * self.tile_size * self.resolution
                tile_y_world = ty * self.tile_size * self.resolution
                
                ox = int((tile_x_world - (xmin - margin)) / self.resolution)
                oy = int((tile_y_world - (ymin - margin)) / self.resolution)
                
                # Reconstruct this tile
                d = self.tiles[(tx, ty)]
                cur = d['pyr'][-1]
                for i in range(self.band_num - 1, -1, -1):
                    cur = cv2.add(d['pyr'][i], 
                                 cv2.pyrUp(cur, dstsize=(d['pyr'][i].shape[1], 
                                                        d['pyr'][i].shape[0])))
                
                tile_h, tile_w = cur.shape[:2]
                
                # Place in canvas (with bounds checking)
                y1 = max(0, oy)
                y2 = min(canvas_h, oy + tile_h)
                x1 = max(0, ox)
                x2 = min(canvas_w, ox + tile_w)
                
                sy1 = y1 - oy
                sy2 = sy1 + (y2 - y1)
                sx1 = x1 - ox
                sx2 = sx1 + (x2 - x1)
                
                if sy2 > sy1 and sx2 > sx1:
                    canvas[y1:y2, x1:x2] = np.clip(cur[sy1:sy2, sx1:sx2], 0, 255)
            
            return canvas

    def feed(self, frame, pose_matrix, camera_matrix, plane_height=0.0):
        """
        Add a new frame to the map with alignment verification.
        
        Args:
            frame: Input image (BGR)
            pose_matrix: 4x4 camera pose in NED frame (Z points down)
            camera_matrix: 3x3 camera intrinsic matrix
            plane_height: Z-coordinate of ground plane (0 for ground level)
        
        Note: In NED frame, altitude should be NEGATIVE (below origin).
        """
        if self.paused: 
            return False
            
        h, w = frame.shape[:2]
        R = pose_matrix[:3, :3]  # Rotation matrix
        t = pose_matrix[:3, 3]   # Translation vector (camera position)
        
        # === STEP 1: QUALITY ASSESSMENT ===
        # Compute viewing angle (angle from nadir)
        # Camera's +Z axis in world frame (should point down in NED)
        camera_down_vector = R @ np.array([0, 0, 1])
        nadir_vector = np.array([0, 0, 1])  # Down in NED
        viewing_angle_rad = np.arccos(np.clip(np.dot(camera_down_vector, nadir_vector), -1, 1))
        viewing_angle_deg = np.degrees(viewing_angle_rad)
        
        quality_score = FrameQualityAssessor.compute_quality_score(
            frame, t[2], viewing_angle_deg
        )
        
        if quality_score < self.min_quality:
            self.frames_rejected_quality += 1
            print(f"⚠ Frame rejected: Low quality ({quality_score:.2f} < {self.min_quality})")
            return False
        
        # === STEP 2: GEOMETRIC PROJECTION ===
        # Image corner points
        pts_src = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
        
        # Camera coordinates from image coordinates
        u = np.array([0, w-1, w-1, 0], np.float32)
        v = np.array([0, 0, h-1, h-1], np.float32)
        
        # Unproject to normalized camera coordinates
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        
        # Ray directions in camera frame
        ray_camera = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=1)
        ray_camera /= np.linalg.norm(ray_camera, axis=1, keepdims=True)
        
        # Transform rays to world frame (NED: X=North, Y=East, Z=Down)
        ray_world = (R @ ray_camera.T).T
        
        # Intersect with ground plane
        lams = (plane_height - t[2]) / ray_world[:, 2]
        
        # Check for rays pointing away from ground
        if np.any(lams < 0):
            print("⚠ Frame rejected: Rays don't intersect ground plane")
            return False
        
        # World coordinates of corner points
        pts_world = t + lams[:, None] * ray_world
        pts_metric = pts_world[:, :2]  # X, Y only
        
        # Calculate coverage area
        xmin, xmax = pts_metric[:, 0].min(), pts_metric[:, 0].max()
        ymin, ymax = pts_metric[:, 1].min(), pts_metric[:, 1].max()
        
        # Convert to pixel coordinates in map
        pts_pixels = pts_metric / self.resolution
        
        # Order points consistently
        def order_quad(pts):
            c = np.mean(pts, axis=0)
            ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            return pts[np.argsort(ang)]
        
        pts_pixels = order_quad(pts_pixels.astype(np.float32))
        
        # Shift to local coordinate system for warping
        local_offset = pts_pixels.min(axis=0)
        pts_pixels_local = pts_pixels - local_offset
        
        # Calculate output size with padding
        out_w = int(np.ceil(pts_pixels_local[:, 0].max())) + 10
        out_h = int(np.ceil(pts_pixels_local[:, 1].max())) + 10
        
        # Initial warp based on pose
        H_pose = cv2.getPerspectiveTransform(pts_src, pts_pixels_local)
        warped = cv2.warpPerspective(frame, H_pose, (out_w, out_h))
        
        # === STEP 3: ALIGNMENT VERIFICATION ===
        H_final = H_pose  # Default to pose-based transform
        alignment_verified = False
        
        if self.enable_alignment_check:
            # Get existing map region for matching
            map_region = self._get_map_region(xmin, xmax, ymin, ymax)
            
            if map_region is not None and map_region.sum() > 0:  # Map exists in this area
                # Resize for faster matching if needed
                max_dim = 1200
                if max(warped.shape[:2]) > max_dim:
                    scale = max_dim / max(warped.shape[:2])
                    warped_small = cv2.resize(warped, None, fx=scale, fy=scale)
                    map_small = cv2.resize(map_region, None, fx=scale, fy=scale)
                else:
                    warped_small = warped
                    map_small = map_region
                
                # Match features
                match_result = self.feature_matcher.match_and_align(warped_small, map_small)
                
                if match_result['success'] and match_result['inlier_ratio'] >= 0.5:
                    # Good alignment! Refine the transform
                    H_refine = match_result['homography']
                    
                    # Scale H_refine back if we downsampled
                    if max(warped.shape[:2]) > max_dim:
                        scale_inv = 1.0 / scale
                        S = np.diag([scale_inv, scale_inv, 1.0])
                        S_inv = np.diag([scale, scale, 1.0])
                        H_refine = S @ H_refine @ S_inv
                    
                    # Apply refinement to original warped image
                    warped = cv2.warpPerspective(warped, H_refine, (out_w, out_h))
                    alignment_verified = True
                    
                    print(f"✓ Alignment verified: {match_result['inliers']} inliers "
                          f"({match_result['inlier_ratio']:.1%}) - Quality: {quality_score:.2f}")
                
                elif match_result['inlier_ratio'] < 0.3:
                    # Very poor alignment - reject frame
                    self.frames_rejected_alignment += 1
                    print(f"✗ Frame rejected: Poor alignment "
                          f"({match_result['inliers']} inliers, "
                          f"{match_result['inlier_ratio']:.1%} ratio)")
                    return False
                else:
                    # Marginal alignment - use pose-based but warn
                    print(f"⚠ Marginal alignment ({match_result['inlier_ratio']:.1%}), "
                          f"using pose-based transform")
        
        # === STEP 4: BLEND INTO MAP ===
        wmask = cv2.warpPerspective(self._get_weight_mask(frame.shape), H_pose, (out_w, out_h))
        
        # Boost weight for high-quality, aligned frames
        if alignment_verified:
            wmask *= 1.2
        wmask *= quality_score  # Weight by quality
        
        # Create pyramids for blending
        pyr_img = self._create_laplace_pyr(warped.astype(np.float32))
        pyr_w = [wmask]
        for _ in range(self.band_num):
            pyr_w.append(cv2.pyrDown(pyr_w[-1]))
        
        # Determine which tiles this frame touches
        def tile_index(v):
            return int(np.floor(v / (self.resolution * self.tile_size)))
        
        tminx, tmaxx = tile_index(xmin), tile_index(xmax)
        tminy, tmaxy = tile_index(ymin), tile_index(ymax)
        
        # Update tiles
        with self.lock:
            for tx in range(tminx, tmaxx + 1):
                for ty in range(tminy, tmaxy + 1):
                    key = (tx, ty)
                    
                    # Initialize tile if needed
                    if key not in self.tiles:
                        self.tiles[key] = {
                            'pyr': [np.zeros((self.tile_size // (2**i), self.tile_size // (2**i), 3), 
                                           np.float32) for i in range(self.band_num + 1)],
                            'w': [np.zeros((self.tile_size // (2**i), self.tile_size // (2**i)), 
                                         np.float32) for i in range(self.band_num + 1)]
                        }
                    
                    # Calculate tile position in world coordinates
                    tile_x_world = tx * self.tile_size * self.resolution
                    tile_y_world = ty * self.tile_size * self.resolution
                    
                    # Offset from world to warped image coordinates
                    sx = int((tile_x_world - xmin) / self.resolution)
                    sy = int((tile_y_world - ymin) / self.resolution)
                    
                    # Blend at each pyramid level
                    for i in range(self.band_num + 1):
                        scale = 2 ** i
                        ts = self.tile_size // scale
                        lx, ly = sx // scale, sy // scale
                        
                        # Calculate valid region in warped image
                        x0 = max(0, lx)
                        y0 = max(0, ly)
                        x1 = min(lx + ts, pyr_img[i].shape[1])
                        y1 = min(ly + ts, pyr_img[i].shape[0])
                        
                        w0, h0 = x1 - x0, y1 - y0
                        if w0 <= 0 or h0 <= 0:
                            continue
                        
                        # Calculate position in tile
                        dx, dy = x0 - lx, y0 - ly
                        
                        # Extract regions
                        img_patch = pyr_img[i][y0:y1, x0:x1]
                        wt_patch = pyr_w[i][y0:y1, x0:x1]
                        
                        tile_img = self.tiles[key]['pyr'][i][dy:dy+h0, dx:dx+w0]
                        tile_w = self.tiles[key]['w'][i][dy:dy+h0, dx:dx+w0]
                        
                        # IMPROVED BLENDING: Weight-based accumulation instead of replacement
                        # This provides smoother transitions and better handling of overlaps
                        new_w = tile_w + wt_patch
                        mask = new_w > 1e-6  # Small threshold to avoid division by zero
                        
                        if mask.any():
                            # Create 3-channel version of weights for color blending
                            tile_w_3d = tile_w[:, :, np.newaxis]
                            wt_patch_3d = wt_patch[:, :, np.newaxis]
                            new_w_3d = new_w[:, :, np.newaxis]
                            
                            # Weighted average where overlap exists
                            tile_img[mask] = (
                                (tile_img[mask] * tile_w_3d[mask] + 
                                 img_patch[mask] * wt_patch_3d[mask]) / 
                                new_w_3d[mask]
                            ).astype(np.float32)
                            
                            # Update weight map
                            tile_w[mask] = new_w[mask]
        
        self.frames_processed += 1
        
        if self.frames_processed % 10 == 0:
            print(f"\n📊 Stats: {self.frames_processed} processed, "
                  f"{self.frames_rejected_quality} rejected (quality), "
                  f"{self.frames_rejected_alignment} rejected (alignment)")
        
        return True

    def render_map(self, quality_lvl=0):
        """
        Render the complete map by reconstructing from pyramids.
        
        Args:
            quality_lvl: pyramid level to render (0 = full quality)
        """
        with self.lock:
            if not self.tiles:
                return None
            
            ks = list(self.tiles.keys())
            minx = min(k[0] for k in ks)
            maxx = max(k[0] for k in ks)
            miny = min(k[1] for k in ks)
            maxy = max(k[1] for k in ks)
            
            ts = self.tile_size // (2 ** quality_lvl)
            canvas = np.zeros(((maxy - miny + 1) * ts, (maxx - minx + 1) * ts, 3), np.uint8)
            
            for (tx, ty), d in self.tiles.items():
                # Reconstruct from pyramid
                cur = d['pyr'][-1]
                for i in range(self.band_num - 1, quality_lvl - 1, -1):
                    cur = cv2.add(d['pyr'][i], 
                                 cv2.pyrUp(cur, dstsize=(d['pyr'][i].shape[1], 
                                                        d['pyr'][i].shape[0])))
                
                # Place in canvas
                ox = (tx - minx) * ts
                oy = (ty - miny) * ts
                canvas[oy:oy + cur.shape[0], ox:ox + cur.shape[1]] = np.clip(cur, 0, 255)
        
        return canvas

# ================= NETWORK UTILITIES =================
def recv_all(sock, n):
    """Receive exactly n bytes from socket."""
    data = b''
    while len(data) < n:
        p = sock.recv(n - len(data))
        if not p:
            return None
        data += p
    return data

def quat_to_rot_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def handle_conn(conn, mapper, K):
    """Handle incoming connection and process frames."""
    print("Client connected")
    
    while True:
        # Read message type
        t = recv_all(conn, 1)
        if not t or t == b'S':
            break
        
        # Read pose data
        p = recv_all(conn, 56)  # 7 doubles: tx, ty, tz, qx, qy, qz, qw
        l = recv_all(conn, 4)   # image length
        
        if not p or not l:
            break
        
        tx, ty, tz, qx, qy, qz, qw = struct.unpack('<ddddddd', p)
        L = struct.unpack('<I', l)[0]
        
        # Read and decode image
        img = recv_all(conn, L)
        if not img:
            break
        
        frame = cv2.imdecode(np.frombuffer(img, np.uint8), 1)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Build pose matrix
        R = quat_to_rot_matrix([qx, qy, qz, qw])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        
        # Feed to mapper
        mapper.feed(frame, T, K)
    
    conn.close()
    print("Client disconnected")

# ================= SERVER =================
if __name__ == "__main__":
    # DJI M3T camera parameters
    cam_params = [3000, 4000, 1800, 1800, 1500, 2000]
    K = np.array([
        [cam_params[2], 0, cam_params[4]],
        [0, cam_params[3], cam_params[5]],
        [0, 0, 1]
    ])
    
    print("=" * 60)
    print("IMPROVED AERIAL MAPPER v2.0")
    print("=" * 60)
    print("\nCamera Intrinsics:")
    print(K)
    
    # Create mapper with alignment checking enabled
    mapper = MultiBandMap2D(
        resolution=0.5,          # 50cm per pixel
        band_num=2,              # Pyramid levels
        tile_size=512,           # Tile size
        enable_alignment_check=True,  # Enable feature-based verification
        min_quality=0.3          # Minimum quality threshold
    )
    
    print(f"\nMapper Configuration:")
    print(f"  Resolution: {mapper.resolution}m/pixel")
    print(f"  Alignment Check: {'Enabled' if mapper.enable_alignment_check else 'Disabled'}")
    print(f"  Min Quality: {mapper.min_quality}")
    
    # Start server
    s = socket.socket()
    s.bind(("0.0.0.0", 5005))
    s.listen(5)
    print("\n" + "=" * 60)
    print("Server listening on port 5005...")
    print("Controls: [P] Pause/Play | [S] Save | [Q] Quality Toggle | [ESC] Exit")
    print("=" * 60 + "\n")
    
    # Accept connections in background
    def accept_loop():
        while True:
            conn, addr = s.accept()
            print(f"\n🔗 Connection from {addr}")
            threading.Thread(target=handle_conn, args=(conn, mapper, K), daemon=True).start()
    
    threading.Thread(target=accept_loop, daemon=True).start()
    
    # Display loop
    cv2.namedWindow("Live Map", cv2.WINDOW_NORMAL)
    
    while True:
        img = mapper.render_map(0)
        
        if img is not None:
            display_img = img.copy()
            
            # Add status overlay
            status_y = 30
            if mapper.paused:
                cv2.putText(display_img, "PAUSED", (20, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                status_y += 40
            
            # Stats overlay
            stats_text = f"Frames: {mapper.frames_processed} | Rejected: Q={mapper.frames_rejected_quality} A={mapper.frames_rejected_alignment}"
            cv2.putText(display_img, stats_text, (20, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Live Map", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('p') or key == ord('P'):
            mapper.paused = not mapper.paused
            status = "Paused" if mapper.paused else "Playing"
            print(f"\n⏸ Mapper Status: {status}")
        elif key == ord('q') or key == ord('Q'):
            # Toggle alignment checking
            mapper.enable_alignment_check = not mapper.enable_alignment_check
            status = "Enabled" if mapper.enable_alignment_check else "Disabled"
            print(f"\n🔍 Alignment Check: {status}")
        elif key == ord('s') or key == ord('S'):
            if img is not None:
                filename = f"map_snapshot_{int(time.time())}.png"
                cv2.imwrite(filename, img)
                print(f"\n💾 Map saved to {filename}")
    
    cv2.destroyAllWindows()

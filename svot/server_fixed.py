import cv2
import numpy as np
import socket
import struct
import threading
import time

# ================= MAPPER ENGINE =================
class MultiBandMap2D:
    def __init__(self, resolution=0.05, band_num=2, tile_size=512):
        """
        Multi-band blending mapper for aerial imagery.
        
        Args:
            resolution: meters per pixel in output map
            band_num: number of pyramid levels for blending
            tile_size: size of each map tile in pixels
        """
        self.resolution = resolution
        self.band_num = band_num
        self.tile_size = tile_size
        self.tiles = {}
        self.weight_mask = None
        self.last_frame_shape = None
        self.lock = threading.Lock()
        self.paused = False

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

    def feed(self, frame, pose_matrix, camera_matrix, plane_height=0.0):
        """
        Add a new frame to the map.
        
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
        
        # Image corner points in image coordinates
        pts_src = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
        
        # --- PROJECT IMAGE CORNERS TO GROUND PLANE ---
        # Corner pixel coordinates
        u = np.array([0, w-1, w-1, 0], np.float32)
        v = np.array([0, 0, h-1, h-1], np.float32)
        
        # Unproject to normalized camera coordinates
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        
        # Ray directions in camera frame (normalized)
        ray_camera = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=1)
        ray_camera /= np.linalg.norm(ray_camera, axis=1, keepdims=True)
        
        # Transform rays to world frame (NED: X=North, Y=East, Z=Down)
        ray_world = (R @ ray_camera.T).T
        
        # Intersect rays with ground plane (Z = plane_height)
        lams = (plane_height - t[2]) / ray_world[:, 2]
        
        # Check for rays pointing away from ground
        if np.any(lams < 0):
            print("Warning: Some rays don't intersect ground plane")
            return False
        
        # World coordinates of corner points
        pts_world = t + lams[:, None] * ray_world
        pts_metric = pts_world[:, :2]  # X, Y only
        
        # Calculate coverage area
        xmin, xmax = pts_metric[:, 0].min(), pts_metric[:, 0].max()
        ymin, ymax = pts_metric[:, 1].min(), pts_metric[:, 1].max()
        
        width_m = xmax - xmin
        height_m = ymax - ymin
        gsd = width_m / w  # Ground sample distance (meters per pixel)
        
        print(f"\nFrame: {width_m:.2f}m x {height_m:.2f}m at alt {abs(t[2]):.2f}m, GSD={gsd:.4f}m/px")
        
        # Convert to pixel coordinates in map
        pts_pixels = pts_metric / self.resolution
        
        # FIX 1: Proper point ordering (clockwise from top-left)
        # Top-left, Top-right, Bottom-right, Bottom-left
        def order_quad_proper(pts):
            """Order points consistently as TL, TR, BR, BL."""
            # Sort by Y coordinate to separate top and bottom
            sorted_by_y = pts[np.argsort(pts[:, 1])]
            top_two = sorted_by_y[:2]
            bottom_two = sorted_by_y[2:]
            
            # Sort each pair by X coordinate
            top_sorted = top_two[np.argsort(top_two[:, 0])]  # TL, TR
            bottom_sorted = bottom_two[np.argsort(bottom_two[:, 0])]  # BL, BR
            
            # Return as TL, TR, BR, BL (clockwise)
            return np.array([top_sorted[0], top_sorted[1], 
                           bottom_sorted[1], bottom_sorted[0]], dtype=np.float32)
        
        pts_pixels = order_quad_proper(pts_pixels.astype(np.float32))
        
        # FIX 2: Calculate global position BEFORE shifting to local coordinates
        # This is the key fix - we need to know where in world space this warped image goes
        global_xmin_px = pts_pixels[:, 0].min()
        global_ymin_px = pts_pixels[:, 1].min()
        
        # Now shift to local coordinate system for warping
        local_offset = pts_pixels.min(axis=0)
        pts_pixels_local = pts_pixels - local_offset
        
        # Calculate output size with minimal padding
        out_w = int(np.ceil(pts_pixels_local[:, 0].max())) + 2
        out_h = int(np.ceil(pts_pixels_local[:, 1].max())) + 2
        
        # Warp image to map coordinates
        H = cv2.getPerspectiveTransform(pts_src, pts_pixels_local)
        warped = cv2.warpPerspective(frame, H, (out_w, out_h))
        wmask = cv2.warpPerspective(self._get_weight_mask(frame.shape), H, (out_w, out_h))
        
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
                    
                    # FIX 3: Corrected tile position calculation
                    # Tile starts at this world position (in meters)
                    tile_x_world = tx * self.tile_size * self.resolution
                    tile_y_world = ty * self.tile_size * self.resolution
                    
                    # Convert tile world position to map pixel coordinates
                    tile_x_px = tile_x_world / self.resolution
                    tile_y_px = tile_y_world / self.resolution
                    
                    # Calculate offset from warped image origin to tile origin
                    # warped image starts at global_xmin_px, global_ymin_px
                    # tile starts at tile_x_px, tile_y_px
                    sx = int(tile_x_px - global_xmin_px)
                    sy = int(tile_y_px - global_ymin_px)
                    
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
                        
                        # FIX 4: Proper weighted blending instead of replacement
                        # Accumulate weights
                        new_w = tile_w + wt_patch
                        
                        # Blend where there's any weight
                        mask = new_w > 1e-6
                        if mask.any():
                            # Create 3-channel weight for RGB blending
                            tile_w_3d = tile_w[:, :, np.newaxis]
                            wt_patch_3d = wt_patch[:, :, np.newaxis]
                            new_w_3d = new_w[:, :, np.newaxis]
                            
                            # Weighted average
                            tile_img[mask] = (
                                (tile_img[mask] * tile_w_3d[mask] + 
                                 img_patch[mask] * wt_patch_3d[mask]) / 
                                new_w_3d[mask]
                            ).astype(np.float32)
                            
                            # Update weights
                            tile_w[mask] = new_w[mask]
        
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
    # After rotation: 3000x4000 (width x height)
    
    # CALIBRATION METHOD 1: From sensor specs
    # Sensor: 1/1.3" (~9.88mm width)
    # Focal length: 4.5mm (approx, DJI spec varies)
    # After rotation: 4000 pixels wide
    # fx = focal_length_mm * pixels / sensor_width_mm
    # fx = 4.5 * 4000 / 9.88 ≈ 1822 pixels
    
    # CALIBRATION METHOD 2: From field of view
    # DJI M3T: ~82.9° horizontal FOV (wide lens)
    # fx = image_width / (2 * tan(HFOV/2))
    # fx = 4000 / (2 * tan(41.45°)) = 4000 / 1.776 ≈ 2252 pixels
    
    # RECOMMENDATION: Use the average or calibrate with checkerboard
    # For now, using conservative estimate:
    
    cam_params = [3000, 4000, 2250, 2250, 1500, 2000]  # Adjusted focal length
    
    # Or if you want to stick with your current:
    # cam_params = [3000, 4000, 1800, 1800, 1500, 2000]
    
    K = np.array([
        [cam_params[2], 0, cam_params[4]],
        [0, cam_params[3], cam_params[5]],
        [0, 0, 1]
    ])
    
    print("="*60)
    print("FIXED AERIAL MAPPER")
    print("="*60)
    print("\nCamera Intrinsics:")
    print(K)
    print(f"\nFocal length: fx={K[0,0]:.0f}, fy={K[1,1]:.0f} pixels")
    print(f"Principal point: cx={K[0,2]:.0f}, cy={K[1,2]:.0f}")
    print(f"Image size: {cam_params[0]}x{cam_params[1]}")
    
    # Calculate expected GSD at different altitudes
    print("\nExpected Ground Sample Distance (GSD):")
    for alt in [30, 50, 70, 100]:
        # tan(half_fov) = half_width / focal_length
        half_fov_rad = np.arctan(cam_params[0] / (2 * cam_params[2]))
        ground_width = 2 * alt * np.tan(half_fov_rad)
        gsd = ground_width / cam_params[0]
        print(f"  At {alt}m altitude: GSD = {gsd:.4f} m/px, coverage = {ground_width:.1f}m")
    
    # Create mapper
    # Use resolution that matches your expected GSD
    # If flying at 50m, use resolution ≈ 0.02-0.05 m/px
    mapper = MultiBandMap2D(resolution=0.05, band_num=2, tile_size=512)
    
    print(f"\nMap resolution: {mapper.resolution} m/pixel")
    print("="*60)
    
    # Start server
    s = socket.socket()
    s.bind(("0.0.0.0", 5005))
    s.listen(5)
    print("\nServer listening on port 5005...")
    print("Controls: [P] Pause/Play | [S] Save Image | [ESC] Exit\n")
    
    # Accept connections in background
    def accept_loop():
        while True:
            conn, addr = s.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=handle_conn, args=(conn, mapper, K), daemon=True).start()
    
    threading.Thread(target=accept_loop, daemon=True).start()
    
    # Display loop
    cv2.namedWindow("Live Map", cv2.WINDOW_NORMAL)
    
    while True:
        img = mapper.render_map(0)
        
        if img is not None:
            display_img = img.copy()
            
            if mapper.paused:
                cv2.putText(display_img, "PAUSED", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            cv2.imshow("Live Map", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('p') or key == ord('P'):
            mapper.paused = not mapper.paused
            status = "Paused" if mapper.paused else "Playing"
            print(f"Mapper Status: {status}")
        elif key == ord('s') or key == ord('S'):
            if img is not None:
                filename = f"map_snapshot_{int(time.time())}.png"
                cv2.imwrite(filename, img)
                print(f"Map saved to {filename}")
    
    cv2.destroyAllWindows()

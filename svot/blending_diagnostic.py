import cv2
import numpy as np
import socket
import struct
import threading
import time

# ================= BLENDING DIAGNOSTIC MAPPER =================
class BlendingDiagnosticMapper:
    """Diagnose warping and blending issues."""
    
    def __init__(self, resolution=0.5):
        self.resolution = resolution
        self.map_accumulator = None
        self.map_bounds = None
        self.frame_count = 0
        self.save_individual_warps = True
        
    def feed(self, frame, pose_matrix, camera_matrix, frame_id):
        """Process frame and diagnose warping/blending."""
        h, w = frame.shape[:2]
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]
        
        print(f"\n{'='*60}")
        print(f"FRAME {frame_id} - Warping Diagnostics")
        print(f"{'='*60}")
        print(f"Input image: {w}x{h}")
        print(f"Position: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")
        
        # === PROJECTION ===
        pts_src = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
        
        u = np.array([0, w-1, w-1, 0], np.float32)
        v = np.array([0, 0, h-1, h-1], np.float32)
        
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        
        ray_camera = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=1)
        ray_camera /= np.linalg.norm(ray_camera, axis=1, keepdims=True)
        
        ray_world = (R @ ray_camera.T).T
        lams = (0.0 - t[2]) / ray_world[:, 2]
        
        pts_world = t + lams[:, None] * ray_world
        pts_metric = pts_world[:, :2]
        
        print(f"\nCorner projection (world coordinates):")
        for i, corner in enumerate(['TL', 'TR', 'BR', 'BL']):
            print(f"  {corner}: [{pts_metric[i,0]:8.2f}, {pts_metric[i,1]:8.2f}]")
        
        # Calculate footprint
        xmin, xmax = pts_metric[:, 0].min(), pts_metric[:, 0].max()
        ymin, ymax = pts_metric[:, 1].min(), pts_metric[:, 1].max()
        width_m = xmax - xmin
        height_m = ymax - ymin
        
        print(f"\nFootprint size: {width_m:.2f}m x {height_m:.2f}m")
        print(f"Coverage area: {width_m * height_m:.2f} m²")
        print(f"Ground Sample Distance: {width_m/w:.3f}m/pixel")
        
        # Convert to pixels
        pts_pixels = pts_metric / self.resolution
        
        print(f"\nCorner projection (map pixels):")
        for i, corner in enumerate(['TL', 'TR', 'BR', 'BL']):
            print(f"  {corner}: [{pts_pixels[i,0]:8.1f}, {pts_pixels[i,1]:8.1f}]")
        
        # Order points
        def order_quad(pts):
            c = np.mean(pts, axis=0)
            ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            return pts[np.argsort(ang)]
        
        pts_pixels = order_quad(pts_pixels.astype(np.float32))
        
        # Shift to local coordinates
        local_offset = pts_pixels.min(axis=0)
        pts_pixels_local = pts_pixels - local_offset
        
        out_w = int(np.ceil(pts_pixels_local[:, 0].max())) + 10
        out_h = int(np.ceil(pts_pixels_local[:, 1].max())) + 10
        
        print(f"\nWarped output size: {out_w}x{out_h} pixels")
        print(f"Local offset: [{local_offset[0]:.1f}, {local_offset[1]:.1f}]")
        
        # === CHECK HOMOGRAPHY ===
        H = cv2.getPerspectiveTransform(pts_src, pts_pixels_local)
        
        print(f"\nHomography matrix:")
        print(H)
        
        # Check if homography is reasonable
        det_H = np.linalg.det(H)
        print(f"Determinant: {det_H:.6f}")
        
        if abs(det_H) < 1e-6:
            print("⚠ WARNING: Homography nearly singular!")
        
        # === WARP ===
        warped = cv2.warpPerspective(frame, H, (out_w, out_h))
        
        # Check if warp produced valid output
        valid_pixels = np.sum(warped.sum(axis=2) > 0)
        total_pixels = out_w * out_h
        coverage = valid_pixels / total_pixels * 100
        
        print(f"\nWarped image coverage: {valid_pixels}/{total_pixels} pixels ({coverage:.1f}%)")
        
        if coverage < 50:
            print("⚠ WARNING: Less than 50% of warped image has data!")
        
        # Save individual warped frame
        if self.save_individual_warps:
            debug_img = warped.copy()
            
            # Draw border
            cv2.rectangle(debug_img, (0, 0), (out_w-1, out_h-1), (0, 255, 0), 2)
            
            # Draw corner markers
            for i, pt in enumerate(pts_pixels_local.astype(int)):
                cv2.circle(debug_img, tuple(pt), 10, (0, 0, 255), -1)
                cv2.putText(debug_img, ['TL', 'TR', 'BR', 'BL'][i], 
                           tuple(pt + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
            
            filename = f'debug_warp_{frame_id:03d}.jpg'
            cv2.imwrite(filename, debug_img)
            print(f"💾 Saved: {filename}")
        
        # === ACCUMULATE INTO GLOBAL MAP ===
        # Calculate global position
        global_x = int(xmin / self.resolution)
        global_y = int(ymin / self.resolution)
        
        print(f"\nGlobal map position: ({global_x}, {global_y})")
        
        if self.map_accumulator is None:
            # Initialize map
            self.map_bounds = {
                'xmin': global_x,
                'ymin': global_y,
                'xmax': global_x + out_w,
                'ymax': global_y + out_h
            }
            self.map_accumulator = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            self.map_accumulator[:, :] = warped
            
            print("✓ Initialized map")
        else:
            # Update bounds
            new_xmin = min(self.map_bounds['xmin'], global_x)
            new_ymin = min(self.map_bounds['ymin'], global_y)
            new_xmax = max(self.map_bounds['xmax'], global_x + out_w)
            new_ymax = max(self.map_bounds['ymax'], global_y + out_h)
            
            # Expand map if needed
            if (new_xmin != self.map_bounds['xmin'] or 
                new_ymin != self.map_bounds['ymin'] or
                new_xmax != self.map_bounds['xmax'] or 
                new_ymax != self.map_bounds['ymax']):
                
                new_w = new_xmax - new_xmin
                new_h = new_ymax - new_ymin
                
                print(f"📏 Expanding map from {self.map_accumulator.shape[1]}x{self.map_accumulator.shape[0]} to {new_w}x{new_h}")
                
                new_map = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                
                # Copy old map
                old_x = self.map_bounds['xmin'] - new_xmin
                old_y = self.map_bounds['ymin'] - new_ymin
                new_map[old_y:old_y+self.map_accumulator.shape[0], 
                       old_x:old_x+self.map_accumulator.shape[1]] = self.map_accumulator
                
                self.map_accumulator = new_map
                self.map_bounds = {'xmin': new_xmin, 'ymin': new_ymin, 
                                  'xmax': new_xmax, 'ymax': new_ymax}
            
            # Place warped frame
            offset_x = global_x - self.map_bounds['xmin']
            offset_y = global_y - self.map_bounds['ymin']
            
            print(f"Placing at map offset: ({offset_x}, {offset_y})")
            
            # Check bounds
            y1, y2 = offset_y, offset_y + out_h
            x1, x2 = offset_x, offset_x + out_w
            
            map_h, map_w = self.map_accumulator.shape[:2]
            
            # Clip to map bounds
            sy1 = max(0, -offset_y)
            sx1 = max(0, -offset_x)
            sy2 = min(out_h, map_h - offset_y)
            sx2 = min(out_w, map_w - offset_x)
            
            dy1 = max(0, offset_y)
            dx1 = max(0, offset_x)
            dy2 = min(map_h, offset_y + out_h)
            dx2 = min(map_w, offset_x + out_w)
            
            if dy2 > dy1 and dx2 > dx1 and sy2 > sy1 and sx2 > sx1:
                # Simple alpha blending (50/50 for now)
                existing = self.map_accumulator[dy1:dy2, dx1:dx2]
                incoming = warped[sy1:sy2, sx1:sx2]
                
                # Only blend where both have data
                mask_existing = existing.sum(axis=2) > 0
                mask_incoming = incoming.sum(axis=2) > 0
                mask_overlap = mask_existing & mask_incoming
                
                overlap_pixels = mask_overlap.sum()
                new_pixels = mask_incoming.sum() - overlap_pixels
                
                print(f"Overlap: {overlap_pixels} pixels, New: {new_pixels} pixels")
                
                if overlap_pixels > 0:
                    print(f"⚠ OVERLAP DETECTED - Blending {overlap_pixels} pixels")
                
                # Blend where overlap
                blended = existing.copy()
                if mask_overlap.any():
                    mask_3d = mask_overlap[:, :, np.newaxis]
                    blended[mask_3d] = (existing[mask_3d] * 0.5 + incoming[mask_3d] * 0.5).astype(np.uint8)
                
                # Add new pixels
                mask_new = mask_incoming & ~mask_existing
                if mask_new.any():
                    mask_3d = mask_new[:, :, np.newaxis]
                    blended[mask_3d] = incoming[mask_3d]
                
                self.map_accumulator[dy1:dy2, dx1:dx2] = blended
                print("✓ Frame integrated into map")
            else:
                print("✗ WARNING: Frame outside map bounds!")
        
        self.frame_count += 1
        
        # Save accumulated map
        cv2.imwrite('accumulated_map.jpg', self.map_accumulator)
        print(f"💾 Updated: accumulated_map.jpg")
        
        return True
    
    def get_map(self):
        return self.map_accumulator

# ================= NETWORK UTILITIES =================
def recv_all(sock, n):
    data = b''
    while len(data) < n:
        p = sock.recv(n - len(data))
        if not p:
            return None
        data += p
    return data

def quat_to_rot_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def handle_conn(conn, mapper, K):
    print("Client connected - BLENDING DIAGNOSTIC MODE")
    frame_count = 0
    
    while True:
        t = recv_all(conn, 1)
        if not t or t == b'S':
            break
        
        p = recv_all(conn, 56)
        l = recv_all(conn, 4)
        
        if not p or not l:
            break
        
        tx, ty, tz, qx, qy, qz, qw = struct.unpack('<ddddddd', p)
        L = struct.unpack('<I', l)[0]
        
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
        
        # Process with diagnostics
        mapper.feed(frame, T, K, frame_count)
        
        frame_count += 1
        
        # Process limited frames for diagnosis
        if frame_count >= 5:
            print("\n" + "="*60)
            print("Processed 5 frames - check individual warps and accumulated map")
            print("="*60)
            break
    
    conn.close()
    print("\nClient disconnected")

# ================= MAIN =================
if __name__ == "__main__":
    print("="*60)
    print("BLENDING DIAGNOSTIC MODE")
    print("="*60)
    print("This will:")
    print("  1. Save each warped frame: debug_warp_XXX.jpg")
    print("  2. Show detailed warping info in console")
    print("  3. Create accumulated map: accumulated_map.jpg")
    print("="*60 + "\n")
    
    # Camera parameters
    cam_params = [3000, 4000, 1800, 1800, 1500, 2000]
    K = np.array([
        [cam_params[2], 0, cam_params[4]],
        [0, cam_params[3], cam_params[5]],
        [0, 0, 1]
    ])
    
    print("Camera Intrinsics:")
    print(K)
    
    # Create mapper
    mapper = BlendingDiagnosticMapper(resolution=0.5)
    
    # Start server
    s = socket.socket()
    s.bind(("0.0.0.0", 5005))
    s.listen(1)
    print("\nServer listening on port 5005...")
    print("Waiting for connection...\n")
    
    # Accept connection
    conn, addr = s.accept()
    print(f"Connection from {addr}\n")
    handle_conn(conn, mapper, K)
    
    print("\n✓ Diagnostic complete!")
    print("\nCheck these files:")
    print("  - debug_warp_000.jpg, debug_warp_001.jpg, etc. (individual warped frames)")
    print("  - accumulated_map.jpg (final blended result)")
    print("\nLook for:")
    print("  • Are warped frames distorted?")
    print("  • Do overlap regions align properly?")
    print("  • Are there gaps between frames in accumulated_map.jpg?")

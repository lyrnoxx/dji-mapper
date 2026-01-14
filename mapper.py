import cv2
import numpy as np
import socket
import struct
import threading
import time

# ================= MAPPER ENGINE =================
class MultiBandMap2D:
    def __init__(self, resolution=1.5, band_num=2, tile_size=512):
        self.resolution = resolution
        self.band_num = band_num
        self.tile_size = tile_size
        self.tiles = {}
        self.weight_mask = None
        self.last_frame_shape = None
        self.lock = threading.Lock()
        self.paused = False

    def _get_weight_mask(self, shape):
        if self.weight_mask is not None and self.last_frame_shape == shape:
            return self.weight_mask
        h, w = shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        maxd = np.sqrt(center_x**2 + center_y**2)
        mask = np.clip(1.0 - dist / maxd, 1e-5, 1.0).astype(np.float32)
        self.weight_mask = mask * mask
        self.last_frame_shape = shape
        return self.weight_mask

    def _create_laplace_pyr(self, img):
        pyr, cur = [], img
        for _ in range(self.band_num):
            down = cv2.pyrDown(cur)
            up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            pyr.append(cv2.subtract(cur, up))
            cur = down
        pyr.append(cur)
        return pyr

    def feed(self, frame, pose_matrix, camera_matrix, plane_height=0.0):
        # Skip processing if paused
        if self.paused: 
            return False
            
        t0 = time.time()
        h, w = frame.shape[:2]
        R, t = pose_matrix[:3, :3], pose_matrix[:3, 3]

        pts_src = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32)
        uv_h = np.column_stack((pts_src, np.ones(4)))
        K_inv = np.linalg.inv(camera_matrix)
        ray_camera = (K_inv @ uv_h.T).T
        ray_camera[:,2] *= -1  # nadir
        ray_world = (R @ ray_camera.T).T

        denom = ray_world[:,2]
        valid = np.abs(denom) > 1e-6
        if not np.any(valid): return False
        lams = (plane_height - t[2]) / denom[valid]

        pts_metric = (t + lams[:,None] * ray_world[valid])[:,:2]
        xmin, xmax = pts_metric[:,0].min(), pts_metric[:,0].max()
        ymin, ymax = pts_metric[:,1].min(), pts_metric[:,1].max()

        roi_w = int((xmax - xmin) / self.resolution) + 20
        roi_h = int((ymax - ymin) / self.resolution) + 20
        if roi_w <= 0 or roi_h <= 0 or roi_w > 4000 or roi_h > 4000: return False

        T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
        S = np.array([[1/self.resolution,0,0],[0,1/self.resolution,0],[0,0,1]])
        M = S @ T
        dst_px = cv2.transform(np.array([pts_metric]), M[:2,:3])[0]
        H = cv2.getPerspectiveTransform(pts_src, dst_px.astype(np.float32))
        warped = cv2.warpPerspective(frame, H, (roi_w, roi_h))
        wmask = cv2.warpPerspective(self._get_weight_mask(frame.shape), H, (roi_w, roi_h))

        pyr_img = self._create_laplace_pyr(warped.astype(np.float32))
        pyr_w = [wmask]
        for _ in range(self.band_num):
            pyr_w.append(cv2.pyrDown(pyr_w[-1]))

        tminx = int(np.floor(xmin / (self.resolution*self.tile_size)))
        tmaxx = int(np.ceil (xmax / (self.resolution*self.tile_size)))
        tminy = int(np.floor(ymin / (self.resolution*self.tile_size)))
        tmaxy = int(np.ceil (ymax / (self.resolution*self.tile_size)))

        with self.lock:
            for tx in range(tminx, tmaxx+1):
                for ty in range(tminy, tmaxy+1):
                    key=(tx,ty)
                    if key not in self.tiles:
                        self.tiles[key]={'pyr':[np.zeros((self.tile_size//(2**i),self.tile_size//(2**i),3),np.float32) for i in range(self.band_num+1)],
                                          'w':[np.zeros((self.tile_size//(2**i),self.tile_size//(2**i)),np.float32) for i in range(self.band_num+1)]}
                    twx = tx*self.tile_size*self.resolution
                    twy = ty*self.tile_size*self.resolution
                    sx = int((twx - xmin)/self.resolution)
                    sy = int((twy - ymin)/self.resolution)
                    for i in range(self.band_num+1):
                        scale=2**i
                        ts=self.tile_size//scale
                        lx,ly = sx//scale, sy//scale
                        x0=max(0,lx); y0=max(0,ly)
                        w0=min(lx+ts,pyr_img[i].shape[1])-x0
                        h0=min(ly+ts,pyr_img[i].shape[0])-y0
                        if w0<=0 or h0<=0: continue
                        dx,dy = x0-lx,y0-ly
                        img=pyr_img[i][y0:y0+h0,x0:x0+w0]
                        wt =pyr_w[i][y0:y0+h0,x0:x0+w0]
                        tile=self.tiles[key]['pyr'][i][dy:dy+h0,dx:dx+w0]
                        tilew=self.tiles[key]['w'][i][dy:dy+h0,dx:dx+w0]
                        m=wt>tilew
                        tilew[m]=wt[m]
                        tile[m.repeat(3).reshape(m.shape+(3,))]=img[m.repeat(3).reshape(m.shape+(3,))]
        return True

    def render_map(self, quality_lvl=0):
        with self.lock:
            if not self.tiles: return None
            ks=list(self.tiles)
            minx,maxx=min(k[0] for k in ks),max(k[0] for k in ks)
            miny,maxy=min(k[1] for k in ks),max(k[1] for k in ks)
            ts=self.tile_size//(2**quality_lvl)
            canvas=np.zeros(((maxy-miny+1)*ts,(maxx-minx+1)*ts,3),np.uint8)
            for (tx,ty),d in self.tiles.items():
                cur=d['pyr'][-1]
                for i in range(self.band_num-1,quality_lvl-1,-1):
                    cur=cv2.add(d['pyr'][i],cv2.pyrUp(cur,(d['pyr'][i].shape[1],d['pyr'][i].shape[0])))
                ox,oy=(tx-minx)*ts,(ty-miny)*ts
                canvas[oy:oy+cur.shape[0],ox:ox+cur.shape[1]]=np.clip(cur,0,255)
        return canvas

def recv_all(sock,n):
    data=b''
    while len(data)<n:
        p=sock.recv(n-len(data))
        if not p: return None
        data+=p
    return data

def quat_to_rot_matrix(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def handle_conn(conn,mapper,K):
    while True:
        t=recv_all(conn,1)
        if not t or t==b'S': break
        p=recv_all(conn,56); l=recv_all(conn,4)
        tx,ty,tz,qx,qy,qz,qw=struct.unpack('<ddddddd',p)
        L=struct.unpack('<I',l)[0]
        img=recv_all(conn,L)
        frame=cv2.imdecode(np.frombuffer(img,np.uint8),1)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        R = quat_to_rot_matrix([qx, qy, qz, qw])
        
        #R = quat_to_rot_matrix([qx, qy, qz, qw]) @ np.array([
        #    [0,  -1,  0],
        #    [-1,  0,  0],
        #    [0,  0,  -1]
        #])

        T=np.eye(4); T[:3,:3]=R; T[:3,3]=[tx,ty,tz]
        mapper.feed(frame,T,K)
    conn.close()

# ================= SERVER =================
if __name__=="__main__":
    #cam_params=[2000,1500,1333.33,1333.33,1000,750]
    cam_params = [4000, 3000, 2666.67, 2666.67, 2000, 1500]
    K=np.array([[cam_params[2],0,cam_params[4]],[0,cam_params[3],cam_params[5]],[0,0,1]])
    mapper=MultiBandMap2D()
    
    s=socket.socket(); s.bind(("0.0.0.0",5005)); s.listen(5)
    print("Server listening on port 5005...")
    print("Controls: [P] Pause/Play | [S] Save Image | [ESC] Exit")
    
    threading.Thread(
        target=lambda: handle_conn(s.accept()[0], mapper, K),
        daemon=True
    ).start()

    cv2.namedWindow("Live Map", cv2.WINDOW_NORMAL)
    
    while True:
        img = mapper.render_map(0)
        
        if img is not None:
            display_img = img.copy()
            # Overlay status text if paused
            if mapper.paused:
                cv2.putText(display_img, "PAUSED", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            cv2.imshow("Live Map", display_img)
        
        key = cv2.waitKey(1)
        if key == 27: # ESC
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

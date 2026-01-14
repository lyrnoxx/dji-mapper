import cv2, numpy as np, socket, struct, threading, time

class SimpleMapper:
    def __init__(self, res=1.0):
        self.res, self.tiles, self.lock = res, {}, threading.Lock()

    def feed(self, frame, T, K):
        h, w = frame.shape[:2]
        R, t = T[:3, :3], T[:3, 3]
        # Ray-casting corners to ground (Z=0)
        u, v = np.array([0, w, w, 0]), np.array([0, 0, h, h])
        ray = np.stack([(u-K[0,2])/K[0,0], (v-K[1,2])/K[1,1], -np.ones(4)], axis=1)
        ray_w = (R @ (ray / np.linalg.norm(ray, axis=1, keepdims=True)).T).T
        pts_metric = (t + (-t[2] / ray_w[:,2])[:, None] * ray_w)[:, :2]
        
        # Warp image to local box
        dst_px = (pts_metric - pts_metric.min(axis=0)) / self.res
        H = cv2.getPerspectiveTransform(np.array([[0,0],[w,0],[w,h],[0,h]], np.float32), dst_px.astype(np.float32))
        warped = cv2.warpPerspective(frame, H, (int(dst_px[:,0].max()), int(dst_px[:,1].max())))
        
        # Place on map (simplified tiling)
        with self.lock:
            self.tiles[time.time()] = (pts_metric.min(axis=0), warped)

    def render(self):
        # Placeholder: for minimal code, just returns the last processed tile
        if not self.tiles: return None
        return list(self.tiles.values())[-1][1]

def recv_all(sock,n):
    data=b''
    while len(data)<n:
        p=sock.recv(n-len(data))
        if not p: return None
        data+=p
    return data

def handle_conn(conn,mapper,K):
    while True:
        t=recv_all(conn,1)
        if not t or t==b'S': break
        p=recv_all(conn,56); l=recv_all(conn,4)
        tx,ty,tz,qx,qy,qz,qw=struct.unpack('<ddddddd',p)
        L=struct.unpack('<I',l)[0]
        img=recv_all(conn,L)
        frame=cv2.imdecode(np.frombuffer(img,np.uint8),1)

        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)
        
        #R = quat_to_rot_matrix([qx, qy, qz, qw])
        
        #R = quat_to_rot_matrix([qx, qy, qz, qw]) @ np.array([
        #    [1,  0,  0],
        #    [0,  1,  0],
        #    [0,  0,  1]
        #])

        T=np.eye(4); T[:3,:3]=R; T[:3,3]=[tx,ty,tz]
        mapper.feed(frame,T,K)
    conn.close()


if __name__ == "__main__":
    K = np.array([[2666.67, 0, 2000], [0, 2666.67, 1500], [0, 0, 1]])
    mapper = SimpleMapper()
    
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
            #display_img = cv2.flip(img, 1)
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
import socket
import struct
import time
import os
import threading
import sys

# CONFIG
HOST = '127.0.0.1'
PORT = 5005
RGB_FOLDER = "images-true"
TRAJ_FILE_1 = "pose1.txt"

class DroneReplay(threading.Thread):
    def __init__(self, drone_id, traj_file, image_folder, delay=0.1):
        super().__init__()
        self.drone_id = drone_id
        self.traj_file = traj_file
        self.image_folder = image_folder
        self.delay = delay
        self.sock = None

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((HOST, PORT))
            print(f"[{self.drone_id}] Connected to Ground Station.")
            return True
        except ConnectionRefusedError:
            print(f"[{self.drone_id}] Connection Failed! Is server running?")
            return False

    def run(self):
        if not os.path.exists(self.traj_file):
            print(f"[{self.drone_id}] Error: Trajectory file '{self.traj_file}' missing!")
            return
        
        # Retry connection
        while not self.connect():
            time.sleep(2)

        with open(self.traj_file, 'r') as f:
            lines = f.readlines()

        print(f"[{self.drone_id}] Starting Mission ({len(lines)} frames).")

        for i, line in enumerate(lines):
            if line.startswith('#'):
                continue

            parts = line.strip().split()
            if len(parts) < 8:
                continue

            # New format
            img_id = parts[0]

            try:
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
            except ValueError:
                continue

            # Locate image (no extension in traj.txt)
            img_path = None
            for ext in [".png", ".jpg", ".jpeg", ".JPG"]:
                p = os.path.join(self.image_folder, img_id + ext)
                if os.path.exists(p):
                    img_path = p
                    break

            if img_path is None:
                # Silent skip (expected for dropped frames)
                continue

            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
            except Exception:
                continue

            img_len = len(img_data)

            # Protocol:
            # 1 byte  : 'D'
            # 7 double: tx ty tz qx qy qz qw
            # 4 bytes : image length
            header = struct.pack(
                '<cdddddddI',
                b'D',
                tx, ty, tz,
                qx, qy, qz, qw,
                img_len
            )

            try:
                self.sock.sendall(header + img_data)

                if i % 50 == 0:
                    print(f"[{self.drone_id}] Sent frame {img_id} ({i}/{len(lines)})")

            except BrokenPipeError:
                print(f"[{self.drone_id}] Server disconnected.")
                break
            except Exception as e:
                print(f"[{self.drone_id}] Socket Error: {e}")
                break

            time.sleep(self.delay)

        # End of mission
        try:
            self.sock.sendall(b'S')
            self.sock.close()
            print(f"[{self.drone_id}] Mission Complete.")
        except:
            pass

if __name__ == "__main__":
    if not os.path.exists(RGB_FOLDER):
        print(f"ERROR: '{RGB_FOLDER}' folder not found.")
        sys.exit(1)

    drone1 = DroneReplay("Alpha", TRAJ_FILE_1, RGB_FOLDER, delay=0.5)
    drone1.start()
    drone1.join()

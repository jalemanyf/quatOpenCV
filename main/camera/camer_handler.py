import cv2
from threading import Thread, Lock
import numpy as np

class CameraHandler():
    def __init__(self):
        #self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture('rtsp://quatechnion:Adminphygi123@192.168.1.41/stream1')
        self.cap.set(cv2.CAP_PROP_FPS, 4)
        self.lock = Lock()
        self.running = True
        self.loop_thread = Thread(target=self.loop)
        self.loop_thread.start()
        self.frame = np.zeros(640)
        self.ret = False

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()
    
    def loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if self.running:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def finish(self):
        if self.running:
            self.running = False
            self.loop_thread.join()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def __del__(self):
        self.finish()


import cv2
import threading


### We use a async video capture to avoid blocking the main thread, which can cause issues with the GUI and other processes.
class VideoCaptureAsync:
    def __init__(self, src: str = "0"):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.cap.read()
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self.ret = False
                break
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def set(self, prop_id: int, value: float):
        return self.cap.set(prop_id, value)

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

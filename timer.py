import time


class Timer:
    def __init__(self):
        self.timer = time.time()

    def start(self):
        self.timer = time.time()

    def measure(self, message=""):
        end = time.time()
        print(message + ": " + str(end - self.timer) + "s")
        self.start()

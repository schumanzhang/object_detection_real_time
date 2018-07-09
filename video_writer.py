import cv2

class VideoWriter(object):

    def __init__(self, path, size):
        self.path = path
        self.size = size
        self.writer = cv2.VideoWriter(self.path, 
                         cv2.VideoWriter_fourcc('F','M','P','4'), 
                         20.0, self.size, True)

    def __call__(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()




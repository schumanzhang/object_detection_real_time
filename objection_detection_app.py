import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

from utils.webcam import FPS, WebcamVideoStream
from queue import Queue
from threading import Thread
from analytics.tracking import ObjectTracker
from video_writer import VideoWriter
from detect_object import detect_objects

CWD_PATH = os.getcwd()

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_MODEL = os.path.join(CWD_PATH, 'detection', 'tf_models', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_VIDEO = os.path.join(CWD_PATH, 'input.mp4')


def worker(input_q, output_q):
    # load the frozen tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame, sess, detection_graph))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=1280, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=720, help='Height of the frames in the video stream.')
    args = parser.parse_args()

    input_q = Queue(5)
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    writer = VideoWriter('output.mp4', (args.width, args.height))

    '''
    stream = cv2.VideoCapture(0)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    '''

    fps = FPS().start()
    object_tracker = ObjectTracker(path='./', file_name='report.csv')
    while True:
        frame = video_capture.read()
        # (ret, frame) = stream.read()
        fps.update()

        if fps.get_numFrames() % 2 != 0:
            continue

        # put data into the input queue 
        input_q.put(frame)

        t = time.time()

        if output_q.empty():
            pass  # fill up queue
        else:
            data = output_q.get()
            context = {'frame': frame, 'class_names': data['class_names'], 'rec_points': data['rect_points'], 'class_colors': data['class_colors'],
                        'width': args.width, 'height': args.height, 'frame_number': fps.get_numFrames()}
            new_frame = object_tracker(context)
            writer(new_frame)
            cv2.imshow('Video', new_frame)

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    writer.close()
    cv2.destroyAllWindows()

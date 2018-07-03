import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

from utils.webcam import FPS, WebcamVideoStream, draw_boxes_and_labels
from queue import Queue
from threading import Thread
from utils import label_map_util
from analytics.tracking import ObjectTracker

CWD_PATH = os.getcwd()

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_MODEL = os.path.join(CWD_PATH, 'detection', 'tf_models', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_VIDEO = os.path.join(CWD_PATH, 'input.mp4')

print('cv2', cv2.__version__)

PATH_TO_LABELS = os.path.join(CWD_PATH, 'detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# pass in image_np, returns 
def detect_objects(image_np, sess, detection_graph):

    image_np_expanded = np.expand_dims(image_np, axis=0)
    # print('image_np_expanded')
    # print(image_np)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Do the detection/model prediction here
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    
    # print('num_detections:', num_detections)
    # Visualization of the results of a detection.
    # print('classes:', classes)
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )

    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)

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

    '''
    stream = cv2.VideoCapture(0)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    '''

    fps = FPS().start()
    object_tracker = ObjectTracker()
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
                        'width': args.width, 'height': args.height}
            new_frame = object_tracker(context)
            cv2.imshow('Video', new_frame)

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()

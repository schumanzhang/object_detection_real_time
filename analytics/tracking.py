from collections import defaultdict
import csv
import cv2
import os

# tracks and tallys what's present at any particular time on screen
# puts stats in a csv file in another directory
# [['person: 98%'], ['book: 55%'], ['book: 50%']]
class ObjectTracker(object):
    def __init__(self, path, file_name):
        self.class_counts = {}
        self.occupancy = False
        self.fp = open(os.path.join(path, file_name), 'w')
        self.writer = csv.DictWriter(self.fp, fieldnames=['frame', 'detections'])
        self.writer.writeheader()
        self.prev = None


    def update_class_counts(self, class_names):
        frame_counts = defaultdict(int)
        for item in class_names:
            count_item = item[0].split(':')[0]
            frame_counts[count_item] += 1

        # sort this dictionary?
        self.class_counts = frame_counts

    def update_person_status(self, class_names):
        for item in class_names:
            if item[0].split(':')[0] == 'person':
                self.occupancy = True
                return
        self.occupancy = False                


    def write_to_report(self, frame_number):
        self.writer.writerow({'frame': frame_number, 'detections': self.class_names})


    def __call__(self, context):
        self.update_class_counts(context['class_names'])
        self.update_person_status(context['class_names'])
        frame = context['frame']
        font = cv2.FONT_HERSHEY_SIMPLEX
        for point, name, color in zip(context['rec_points'], context['class_names'], context['class_colors']):

            cv2.rectangle(frame, (int(point['xmin'] * context['width']), int(point['ymin'] * context['height'])),
                            (int(point['xmax'] * context['width']), int(point['ymax'] * context['height'])), color, 3)
            cv2.rectangle(frame, (int(point['xmin'] * context['width']), int(point['ymin'] * context['height'])),
                            (int(point['xmin'] * context['width']) + len(name[0]) * 6,
                            int(point['ymin'] * context['height']) - 10), color, -1, cv2.LINE_AA)
            cv2.putText(frame, name[0], (int(point['xmin'] * context['width']), int(point['ymin'] * context['height'])), font,
                        0.3, (0, 0, 0), 1)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, ("Room occupied: {occupied}".format(occupied=self.occupancy)), (30, 30),
                    font, 0.6, (255, 255, 255), 1)

        if len(list(self.class_counts.keys())) > 0:
            key_1 = str(list(self.class_counts.keys())[0])
            cv2.putText(frame, (key_1 + ':' + str(self.class_counts[key_1])), (int(frame.shape[1] * 0.85), 30), font, 0.6, (255, 255, 255), 1)

        self.write_to_report(context['frame_number'])

        return frame
    
#!/usr/bin/env python3

import os
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


class ObjectDetector:
    PATH_TO_CKPT = "object_detection_models/ssd_resnet50/checkpoint/ckpt-0"
    PATH_TO_CFG = "object_detection_models/ssd_resnet50/pipeline.config"
    PATH_TO_LABELS = "object_detection/data/mscoco_label_map.pbtxt"

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)
        tf.compat.v1.enable_eager_execution()

        self.detection_model = None
        self.category_index = None

        self.load_and_build_detection_model()

    def load_and_build_detection_model(self):
        model_config = config_util.get_configs_from_pipeline_file(self.PATH_TO_CFG)
        self.detection_model = model_builder.build(model_config=model_config['model'], is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(self.PATH_TO_CKPT).expect_partial()
        # ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)

    @tf.function
    def _detect_fn(self, image):
        image, shapes_detected = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes_detected)
        detections_detected = self.detection_model.postprocess(prediction_dict, shapes_detected)

        return detections_detected, prediction_dict, tf.reshape(shapes_detected, [-1])

    def _get_detections_tf2(self, frame_np):
        input_tensor = tf.convert_to_tensor(np.expand_dims(frame_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = self._detect_fn(input_tensor)

        return detections

    def get_formatted_detections(self, image):
        detections = self._get_detections_tf2(image)
        formatted_detections = self.format_detections(detections)

        return formatted_detections

    def get_name_of_detected_class(self, class_id):
        class_name = self.category_index[class_id]['name']
        return class_name

    @staticmethod
    def format_detections(detections):
        formatted_detections = {'detection_boxes': detections['detection_boxes'][0].numpy(),
                                'detection_classes': (detections['detection_classes'][0].numpy() + 1).astype(int),
                                'detection_scores': detections['detection_scores'][0].numpy()}

        return formatted_detections

    def visualize_detections(self, frame_np, formatted_detections):
        image_np_with_detections = frame_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            formatted_detections['detection_boxes'],
            formatted_detections['detection_classes'],
            formatted_detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        return image_np_with_detections


"""
#################################################################################
"""

def main():
    detector = ObjectDetector()

    # cap = cv2.VideoCapture(0)
    frame_counter = 0

    while True:

        # Read frame from camera
        # ret, image_np = cap.read()

        image_np = cv2.imread('/home/szymon/catkin_ws/src/camera_node_v3/src/test2.jpg')

        if frame_counter == 0 or frame_counter % 10 == 0:  # capture every 10th frame to lower fps (GPU constraint)

            detections = detector.get_formatted_detections(image_np)

            image_np_with_detections = detector.visualize_detections(image_np, detections)

            frame_counter = 1

            # Display output
            cv2.imshow('object detection', image_np_with_detections)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            frame_counter += 1
            continue

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



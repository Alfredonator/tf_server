#!/usr/bin/env python3

import cv2
from aruco_detector import ArucoDetector
from client_handler import ClientHandler
from object_detector import ObjectDetector
from utils import Utils


def main_client():
    client_handler = ClientHandler()

    object_detector = ObjectDetector()
    box_detector = ArucoDetector()

    try:
        while True:
            color_frame = client_handler.receive_color_frame()
            depth_frame = client_handler.receive_depth_frame()
            camera_info = client_handler.receive_camera_info()

            # get object and box detections
            object_detections = object_detector.get_formatted_detections(color_frame)
            box_detections = box_detector.get_detected_boxes(color_frame)

            # visualize detections
            image_with_detections = object_detector.visualize_detections(color_frame, object_detections)
            cv2.imshow('object detection', image_with_detections)
            cv2.waitKey(3)

            # get 3D objects and 3D boxes list
            objects_3d_list = Utils.get_objects_3d_coordinates(object_detections, depth_frame, camera_info)
            box_3d_list = Utils.get_boxes_3d_coordinates(box_detections, depth_frame, camera_info)

            # add all 3D objects and all 3D boxes back to camera_node
            client_handler.send_detected_objects_array(objects_3d_list)
            client_handler.send_detected_boxes_array(box_3d_list)

    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        del object_detector
        del box_detector
        del client_handler


if __name__ == '__main__':
    main_client()

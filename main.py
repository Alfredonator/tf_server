#!/usr/bin/env python3

import socket
import cv2
import numpy as np
from PIL import Image

# from aruco_detector import ArucoDetector
from object_detector import ObjectDetector
from utils import Utils


class ClientHandler:
    def __init__(self):
        self.local_ip = "127.0.0.1"
        self.local_port = 20033
        buffer_size_Mb = 1
        self.buffer_size = buffer_size_Mb * 1024 * 1024

        self.udp_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.udp_server_socket.bind((self.local_ip, self.local_port))
        print("UDP server up and listening")

        # while True:
        #     message_and_address_pair = self.udp_server_socket.recvfrom(self.buffer_size)
        #     message = message_and_address_pair[0]
        #
        #     jpg_as_np = np.frombuffer(message, dtype=np.uint8)
        #     image_buffer = cv2.imdecode(jpg_as_np, flags=1)
        #
        #     cv2.imshow('received_image', image_buffer)
        #     cv2.waitKey(0)

    def send_detected_objects_array(self, objects):
        # TODO: Get object list and serialize to JSON
        pass

    def send_detected_boxes_array(self, boxes):
        # TODO: Get boxes list and serialize to JSON
        pass

    def receive_color_frame(self):
        image_buffer = cv2.imdecode(self._receive_frame(), flags=cv2.IMREAD_ANYDEPTH)

        return image_buffer

    def receive_depth_frame(self):
        image_buffer = cv2.imdecode(self._receive_frame(), flags=1)

        return image_buffer

    def _receive_frame(self):
        message_and_address_pair = self.udp_server_socket.recvfrom(self.buffer_size)
        message = message_and_address_pair[0]

        jpg_as_np = np.frombuffer(message, dtype=np.uint8)

        return jpg_as_np


def main_no_client():
    object_detector = ObjectDetector()
    # box_detector = ArucoDetector()

    frame_counter = 0

    while True:
        image_np = cv2.imread('/home/szymon/catkin_ws/src/camera_node_v3/src/test2.jpg')

        if frame_counter == 0 or frame_counter % 10 == 0:  # capture every 10th frame to lower fps (GPU constraint)
            color_image = image_np

            # get object and box detections
            object_detections = object_detector.get_formatted_detections(color_image)
            # box_detections = box_detector.get_detected_boxes(color_image)

            # visualize detections
            image_with_detections = object_detector.visualize_detections(color_image, object_detections)
            cv2.imshow('object detection', image_with_detections)
            cv2.waitKey(3)

            # get 3D objects and 3D boxes list
            # objects_3d_list = Utils.get_objects_3d_coordinates(object_detections, depth_image, camera_info)
            # box_3d_list = Utils.get_boxes_3d_coordinates(box_detections, depth_image, camera_info)

            # add all 3D objects and all 3D boxes to moveit planning scene
            # TODO: Server.send() instead of updating moveit planning scene
            # self.update_planning_scene_with_detected_objects(objects_3d_list)
            # self.update_planning_scene_with_detected_boxes(box_3d_list)

            frame_counter = 1
            # cv2.imshow('object detection', image_np_with_detections)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            frame_counter += 1
            continue

    cv2.destroyAllWindows()


def main():
    local_ip = "127.0.0.1"
    local_port = 20033
    buffer_size_Mb = 1
    buffer_size = buffer_size_Mb * 1024 * 1024

    msg_from_server = "Hello UDP Client"
    bytes_to_send = str.encode(msg_from_server)

    udp_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_server_socket.bind((local_ip, local_port))
    print("UDP server up and listening")

    while True:
        message_and_address_pair = udp_server_socket.recvfrom(buffer_size)
        message = message_and_address_pair[0]

        jpg_as_np = np.frombuffer(message, dtype=np.uint8)
        image_buffer = cv2.imdecode(jpg_as_np, flags=1)

        cv2.imshow('received_image', image_buffer)
        cv2.waitKey(0)


if __name__ == '__main__':
    main_no_client()
    # image_bytes = cv2.imencode('.jpg', image_np)[1].tobytes()


import json
import socket
import cv2
import numpy as np


class ClientHandler:
    def __init__(self):
        self._client_address_port = ("127.0.0.1", 20034)

        self._local_ip = "127.0.0.1"
        self._local_port = 20033

        buffer_size_mb = 1
        self._buffer_size = buffer_size_mb * 1024 * 1024

        self._udp_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self._udp_server_socket.bind((self._local_ip, self._local_port))

        print("UDP server up and listening")

    def send_detected_objects_array(self, objects):
        objects_dict = {}

        for object_instance in objects:
            objects_dict[object_instance.obj_class] = {}
            objects_dict[object_instance.obj_class]['x'] = object_instance.x_middle
            objects_dict[object_instance.obj_class]['y'] = object_instance.y_middle
            objects_dict[object_instance.obj_class]['z'] = object_instance.z_middle

        json_objects_dict = json.dumps(objects_dict, indent=4).encode()
        self._udp_server_socket.sendto(json_objects_dict, self._client_address_port)

    def send_detected_boxes_array(self, boxes):
        boxes_dict = {}

        for box_instance in boxes:
            boxes_dict[box_instance.container_name] = {}
            boxes_dict[box_instance.container_name]['x'] = box_instance.x
            boxes_dict[box_instance.container_name]['y'] = box_instance.y
            boxes_dict[box_instance.container_name]['z'] = box_instance.z

        json_objects_dict = json.dumps(boxes_dict, indent=4).encode()
        self._udp_server_socket.sendto(json_objects_dict, self._client_address_port)

    def receive_color_frame(self):
        image_buffer = cv2.imdecode(self._receive_frame(), flags=1)

        return image_buffer


    def receive_depth_frame(self):
        image_buffer = cv2.imdecode(self._receive_frame(), flags=cv2.IMREAD_ANYDEPTH)

        return image_buffer

    def _receive_frame(self):
        message_and_address_pair = self._udp_server_socket.recvfrom(self._buffer_size)
        message = message_and_address_pair[0]

        jpg_as_np = np.frombuffer(message, dtype=np.uint8)

        return jpg_as_np

    def receive_camera_info(self):
        msg_from_client = self._udp_server_socket.recvfrom(self._buffer_size)
        json_ = msg_from_client[0].decode()
        dict_ = json.loads(json_)

        return dict_

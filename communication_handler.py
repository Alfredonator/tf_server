import base64
import json
import socket
from io import BytesIO
import cv2
from PIL import Image

from aruco_detector import ArucoDetector
from object_detector import ObjectDetector
from utils import Utils


class CommunicationUtils:
    @staticmethod
    def create_detected_objects_dict(objects):
        objects_dict = {}

        for object_instance in objects:
            objects_dict[object_instance.obj_class] = {}
            objects_dict[object_instance.obj_class]['x'] = object_instance.x_middle
            objects_dict[object_instance.obj_class]['y'] = object_instance.y_middle
            objects_dict[object_instance.obj_class]['z'] = object_instance.z_middle

        return objects_dict

    @staticmethod
    def create_detected_boxes_dict(boxes):
        boxes_dict = {}

        for box_instance in boxes:
            boxes_dict[box_instance.container_name] = {}
            boxes_dict[box_instance.container_name]['x'] = box_instance.x
            boxes_dict[box_instance.container_name]['y'] = box_instance.y
            boxes_dict[box_instance.container_name]['z'] = box_instance.z

        return boxes_dict

    @staticmethod
    def decode(received):
        load = json.loads(received.decode())
        # imdata = base64.b64decode(load['image'])
        # im = Image.open(BytesIO(imdata))
        # im.show()

        color_frame = base64.b64decode(load['color_frame'])
        depth_frame = base64.b64decode(load['depth_frame'])
        camera_info = base64.b64decode(load['camera_info'])

        return color_frame, depth_frame, camera_info

    @staticmethod
    def encode(objects_list, boxes_list):
        dict_ = {
            "objects_list": CommunicationUtils.create_detected_objects_dict(objects_list),
            "boxes_list": CommunicationUtils.create_detected_boxes_dict(boxes_list)
        }

        return json.dumps(dict_, indent=4).encode()


class CommunicationHandlerServer:
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 5002
    BUFFER_SIZE = 32768

    def __init__(self):
        self.s = socket.socket()
        self.s.bind((self.SERVER_HOST, self.SERVER_PORT))
        self.s.listen(5)
        print(f"[*] Listening as {self.SERVER_HOST}:{self.SERVER_PORT}")

        self.client_socket = None
        self.address = None

        self.object_detector = ObjectDetector()
        self.box_detector = ArucoDetector()

        self._process_request()

    def __del__(self):
        self.client_socket.close()
        self.s.close()

    def _receive_bytes(self):
        bytes_to_read_length = int(self.client_socket.recv(6).decode())
        print(bytes_to_read_length)
        received = bytearray()

        while True:
            bytes_read = self.client_socket.recv(self.BUFFER_SIZE)
            if not bytes_read or len(received) >= bytes_to_read_length:
                print("breaking")
                break
            else:
                received.extend(bytes_read)
                print(f"appending {bytes_read.decode()}, received length so far {len(received)}")

            if len(received) >= bytes_to_read_length:
                print("breaking")
                break

        return received

    def _send_bytes(self, bytes_):
        self.client_socket.sendall(str(len(bytes_)).encode())

    def _process_request(self):
        while True:
            self.client_socket, self.address = self.s.accept()
            print(f"[+] {self.address} is connected.")

            try:
                while True:
                    received = self._receive_bytes()
                    print(f"str length {len(received)}")
                    (color_frame, depth_frame, camera_info) = CommunicationUtils.decode(received)
                    # TODO: Do stuff with the picture

                    # get object and box detections
                    object_detections = self.object_detector.get_formatted_detections(color_frame)
                    box_detections = self.box_detector.get_detected_boxes(color_frame)

                    # visualize detections
                    image_with_detections = self.object_detector.visualize_detections(color_frame, object_detections)
                    cv2.imshow('object detection', image_with_detections)
                    cv2.waitKey(3)

                    # get 3D objects and 3D boxes list
                    objects_3d_list = Utils.get_objects_3d_coordinates(object_detections, depth_frame, camera_info)
                    box_3d_list = Utils.get_boxes_3d_coordinates(box_detections, depth_frame, camera_info)

                    # TODO: Encode stuff and send back dictionary
                    self._send_bytes(CommunicationUtils.encode(objects_3d_list, box_3d_list))
            except Exception as e:
                print(e)
                print("Client disconnected")


if __name__ == '__main__':
    try:
        CommunicationHandlerServer()
    except KeyboardInterrupt as e:
        print(e, "Shutting down!")
    finally:
        del CommunicationHandlerServer

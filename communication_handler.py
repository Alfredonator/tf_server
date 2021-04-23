import socket
import cv2

from communication_utils import CommunicationUtils
from object_detector import ObjectDetector


class CommunicationHandlerServer:
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 5001
    BUFFER_SIZE = 32768
    DETECTION_THRESHOLD = 0.2

    def __init__(self):
        self.s = socket.socket()
        self.s.bind((self.SERVER_HOST, self.SERVER_PORT))
        self.s.listen(5)
        print(f"[*] Listening as {self.SERVER_HOST}:{self.SERVER_PORT}")

        self.client_socket = None
        self.address = None

        self.object_detector = ObjectDetector()

        self._process_request()

    def __del__(self):
        self.client_socket.close()
        self.s.close()

    def _process_request(self):
        while True:
            self.client_socket, self.address = self.s.accept()
            print(f"[+] {self.address} is connected.")

            try:
                while True:
                    received = self._receive_bytes()
                    color_frame = CommunicationUtils.decode(received)

                    object_detections = self.object_detector.get_formatted_detections(color_frame)

                    # image_with_detections = self.object_detector.visualize_detections(color_frame, object_detections)
                    # cv2.imshow('object detection', image_with_detections)
                    # cv2.waitKey(0)

                    self._send_detections(object_detections)

            except Exception as ex:
                print(ex)
                print("Client disconnected")

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
                # print(f"appending {bytes_read.decode()}, received length so far {len(received)}")

            if len(received) >= bytes_to_read_length:
                print("breaking")
                break

        return received

    def _send_detections(self, object_detections):
        detections_dict = self._create_json(object_detections)

        self.client_socket.sendall(CommunicationUtils.encode(detections_dict))

    def _create_json(self, object_detections):
        detections_dict = {}

        for i in range(len(object_detections['detection_classes'])):
            if object_detections['detection_scores'][i] > self.DETECTION_THRESHOLD:
                detections_dict[object_detections['detection_classes'][i]] = {
                    "detection_score": str(object_detections['detection_scores'][i]),
                    "detection_box": {
                        'x_min': str(object_detections['detection_boxes'][i][1]),
                        'y_min': str(object_detections['detection_boxes'][i][0]),
                        'x_max': str(object_detections['detection_boxes'][i][3]),
                        'y_max': str(object_detections['detection_boxes'][i][2])
                    }
                }

        print(f"Detection_dict\n {detections_dict}")
        return detections_dict


if __name__ == '__main__':
    try:
        CommunicationHandlerServer()
    except KeyboardInterrupt as e:
        print(e, "Shutting down!")
    finally:
        del CommunicationHandlerServer

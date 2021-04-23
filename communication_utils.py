import json
import cv2
import numpy as np


class CommunicationUtils:
    @staticmethod
    def decode(received):
        color_buffer = np.frombuffer(received, dtype=np.uint8)
        color_np = cv2.imdecode(color_buffer, flags=cv2.IMREAD_ANYCOLOR)

        return color_np

    @staticmethod
    def encode(objects_detections):

        return json.dumps(objects_detections, indent=4).encode()

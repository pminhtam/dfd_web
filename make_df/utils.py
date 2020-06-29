from core.xception import xception
from facenet_pytorch import MTCNN
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

detector = MTCNN(device=device)

def crop_image_square(image):
    margin = 0.4
    face_positions = detector.detect(image)
    positions = []
    try:
        for face_position in face_positions[0]:
            # face_position = face_positions[0][i]
            x, y, x2, y2 = face_position
            x, y, w, h = int(x), int(y), int(x2 - x), int(y2 - y)
            image_size = w if w>h else h

            offset = round(margin * (image_size))
            image_size+=2*offset
            print("22:   ", image_size)
            offsetx = int((image_size - w)/2)
            offsety = int((image_size - h)/2)
            print("27   offset : ",offset)
            y0 = round(max(y - offsety, 0))
            x0 = round(max(x - offsetx, 0))
            x1 = round(min(x0 + image_size, image.shape[1]))
            y1 = round(min(y0 + image_size, image.shape[0]))
            # print(y0,y1,x0,x1)
            positions.append((y0,y1,x0,x1))

        # for position in positions:
        #     y0, y1, x0, x1 = position
        #     cv2.rectangle(image, (x0, y0),(x1,y1), (0,0,255), 2)
        #     print(position)
        # plt.imshow(image)
        # plt.show()
    except:
        # print(face_positions)
        return None,None
    return positions

if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("F:/Anh/[IMG] KSTN_CNTT/20161213_120030.jpg"),cv2.COLOR_RGB2BGR)
    # print(image)
    crop_image_square(image)
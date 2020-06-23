from core.xception import xception
from facenet_pytorch import MTCNN
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

detector = MTCNN(device=device)
margin = 0.2
model = xception()
model.load_state_dict(torch.load("../../model/xception/model_pytorch_4.pt",map_location=torch.device('cpu') ))
model.eval()
def detect_cnn(image):
    face_positions = detector.detect(image)
    # try:
    #     # face_position = face_positions[0][0]
    #     face_position = face_positions[0][np.argmax(face_positions[1])]
    #     print(face_position)
    # except:
    #     # print(face_positions)
    #     return None
    # x, y, x2, y2 = face_position
    # x, y, w, h = int(x), int(y), int(x2 - x), int(y2 - y)
    # offsetx = round(margin * (w))
    # offsety = round(margin * (h))
    # y0 = round(max(y - offsety, 0))
    # x1 = round(min(x + w + offsetx, image.shape[1]))
    # y1 = round(min(y + h + offsety, image.shape[0]))
    # x0 = round(max(x - offsetx, 0))
    # #         print(x0,x1,y0,y1)
    # face = image[y0:y1, x0:x1]
    # face = cv2.resize(face,(256,256))
    # face = transforms.ToTensor()(face)
    # print(face)
    # face = face.unsqueeze(0)
    # ressult = model.forward(face)
    # ressult = ressult.squeeze()
    # ressult = ressult.detach().numpy()
    position = []
    fake = []
    try:
        for face_position in face_positions[0]:
            # face_position = face_positions[0][i]
            x, y, x2, y2 = face_position
            x, y, w, h = int(x), int(y), int(x2 - x), int(y2 - y)
            offsetx = round(margin * (w))
            offsety = round(margin * (h))
            y0 = round(max(y - offsety, 0))
            x1 = round(min(x + w + offsetx, image.shape[1]))
            y1 = round(min(y + h + offsety, image.shape[0]))
            x0 = round(max(x - offsetx, 0))
            face = image[y0:y1, x0:x1]
            face = cv2.resize(face,(256,256))
            face = transforms.ToTensor()(face)
            # print(face)
            face = face.unsqueeze(0)
            ressult = model.forward(face)
            ressult = ressult.squeeze()
            ressult = ressult.detach().numpy()
            position.append(face_position)
            fake.append(ressult)
    except:
        # print(face_positions)
        return None
    return position,fake
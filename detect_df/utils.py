from core.xception import xception
from core.efficientnet import EfficientDual
from facenet_pytorch import MTCNN
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

detector = MTCNN(device=device)
margin = 0.2
# model = xception()
# model.load_state_dict(torch.load("../../model/xception/model_pytorch_4.pt",map_location=torch.device('cpu') ))
# model.eval()
model = EfficientDual()
model.load_state_dict(torch.load("../../model/model_dualpytorch3_1.pt", map_location=torch.device('cpu')))
model.eval()
# model.to(device)

def get_model(checkpoint):
    global model
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model.eval()

def draw_label(image, point, label,fake, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    color = (255,0 , 0) if fake > 0.5 else (0, 0, 255)
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), color, cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
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
    # print(face_positions)
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
            face = cv2.resize(face,(128,128))


            f = np.fft.fft2(cv2.cvtColor(face,cv2.COLOR_RGB2GRAY))
            fshift = np.fft.fftshift(f)
            fshift += 1e-8

            magnitude_spectrum = np.log(np.abs(fshift))
            # img = np.concatenate([img,magnitude_spectrum],axis=2)
            # img = np.transpose(img,(2,0,1))
            magnitude_spectrum = cv2.resize(magnitude_spectrum, (128, 128))
            magnitude_spectrum = np.array([magnitude_spectrum])
            print(magnitude_spectrum.shape)
            magnitude_spectrum = np.transpose(magnitude_spectrum, (1,2 , 0))
            magnitude_spectrum = transforms.ToTensor()(magnitude_spectrum)

            # face = transforms.ToTensor()(face)
            face = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])

                                               ])(face)
            # print(face)
            face = face.unsqueeze(0)
            magnitude_spectrum = magnitude_spectrum.unsqueeze(0)
            ressult = model.forward(face,magnitude_spectrum)
            ressult = ressult.squeeze()
            ressult = ressult.detach().numpy()
            position.append(face_position)
            fake.append(ressult)
    except:
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return None,None
    return position,fake
# SIZE = [128,256,512]
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
            print(image_size)
            offset = round(margin * (image_size))
            image_size+=2*offset
            offsetx = int((image_size - w)/2)
            offsety = int((image_size - h)/2)
            print("offset : ",offset)
            y0 = round(max(y - offsety, 0))
            x0 = round(max(x - offsetx, 0))
            x1 = round(min(x0 +image_size, image.shape[1]))
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
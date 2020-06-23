from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import numpy as np
import cv2
import base64
import json
# Create your views here.
import matplotlib.pyplot as plt
from detect_df.utils import detect_cnn
from PIL import Image
from io import BytesIO

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=4, thickness=5):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
def detect_df(request):
    if request.method == "POST":
        try:
            img = request.FILES['image'].read()
        except:
            return render(request, 'detect_df.html')
        bytes = bytearray(img)
        # numpyarray = np.asarray(bytes, dtype=np.uint8)
        # print("numpyarray " ,numpyarray)
        # bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        # bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        # print(bytes)
        # image = 'data:image/jpg;base64,' + base64.b64encode(bytes).decode()
        image_arr = np.asarray(bytes, dtype=np.uint8)
        bgrImage = cv2.cvtColor(cv2.imdecode(image_arr, cv2.IMREAD_COLOR),cv2.COLOR_RGB2BGR)
        position, fake = detect_cnn(bgrImage)
        for i in range(len(position)):
            cv2.rectangle(bgrImage, (int(position[i][0]), int(position[i][1])), (int(position[i][2]), int(position[i][3])), (0, 255, 0), 2)
        # for i in fake:
            label = "{:.3f}, {}".format(float(fake[i]), "Fake" if fake[i] >0.5 else "Real")
            draw_label(bgrImage, (int(position[i][0]), int(position[i][1])), label)

        bgrImage = Image.fromarray(bgrImage, 'RGB')
        data = BytesIO()
        bgrImage.save(data, "JPEG")  # pick your format
        # data64 = base64.b64encode(data.getvalue())
        # bytes_result = bytearray(bgrImage)
        # print(bytes_result)
        image = 'data:image/jpg;base64,' + base64.b64encode(data.getvalue()).decode('utf-8')
        print(position)
        # plt.imshow(bgrImage)
        # plt.show()
        # print(bgrImage)
        return render(request, 'Upload2.html', {"image":image,"result":position})
        # return render(request, 'pages/detect_df.html'),{"image":image}
    return render(request, 'Upload2.html',{"fave_img":False})

def setting_detect(request):
    # Setting config to detect
    if request.method == "POST":
        size = request.POST.get('size')
        model = request.POST.get('model')
        if size != "" or model != "":
            setting = {"size": size,'model': model}
            with open("setting_detect.txt","w") as f:
                json.dump(setting,f)
            return render(request, 'setting_detect.html', setting)
            # return render(request, 'pages/detect_df.html'),{"image":image}
    try:
        with open("setting_detect.txt", "r") as f:
            setting = json.load(f)
    except:
        setting = {}
    return render(request, 'setting_detect.html',setting)

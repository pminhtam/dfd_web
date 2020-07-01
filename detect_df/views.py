from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import numpy as np
import cv2
import base64
import json
# Create your views here.
import matplotlib.pyplot as plt
from detect_df.utils import detect_cnn,draw_label
from PIL import Image
from io import BytesIO

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def detect_df(request):
    if request.method == "POST":
        try:
            img = request.FILES['image'].read()
        except:
            return render(request, 'Upload2.html')
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
        if position == None and fake == None:
            image = 'data:image/jpg;base64,' + base64.b64encode(bytes).decode('utf-8')
            return render(request, 'Upload2.html', {"image": image, "result": position, "show": True})
        for i in range(len(position)):
            color = (0, 255, 0)
            print(color)
            cv2.rectangle(bgrImage, (int(position[i][0]), int(position[i][1])), (int(position[i][2]), int(position[i][3])), color, 2)
        # for i in fake:
            label = "{:.3f}, {}".format(float(fake[i]), "Fake" if fake[i] >0.5 else "Real")
            draw_label(bgrImage, (int(position[i][0]), int(position[i][1])), label,fake=fake[i])

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
        return render(request, 'Upload2.html', {"image":image,"result":position,"show":True})
        # return render(request, 'pages/detect_df.html'),{"image":image}
    return render(request, 'Upload2.html',{"show":False})

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

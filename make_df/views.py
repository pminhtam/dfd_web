from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import numpy as np
import cv2
import base64
import json
# Create your views here.
from make_df.utils import crop_image_square
from make_df.stargan.main import adj_image
from PIL import Image
from io import BytesIO

def make_df(request):
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
        positions = crop_image_square(bgrImage)
        if positions == []:
            image = 'data:image/jpg;base64,' + base64.b64encode(bytes).decode('utf-8')
            return render(request, 'Upload2.html', {"image": image, "result": positions, "show": True})
        for position in positions:
            # color = (0, 255, 0)
            y0, y1, x0, x1 = position
            print(position)
            image_df = adj_image(bgrImage[y0:y1, x0:x1],y1-y0)
            # print(image_df)
            bgrImage[y0:y1, x0:x1] = image_df
            # cv2.rectangle(bgrImage, (int(position[0]), int(position[1])), (int(position[2]), int(position[3])), color, 2)

        bgrImage = Image.fromarray(bgrImage, 'RGB')
        data = BytesIO()
        bgrImage.save(data, "JPEG")  # pick your format
        # data64 = base64.b64encode(data.getvalue())
        # bytes_result = bytearray(bgrImage)
        # print(bytes_result)
        image = 'data:image/jpg;base64,' + base64.b64encode(data.getvalue()).decode('utf-8')
        # print(position)
        # plt.imshow(bgrImage)
        # plt.show()
        # print(bgrImage)
        return render(request, 'Upload2.html', {"image":image,"result":positions,"show":True})
        # return render(request, 'pages/detect_df.html'),{"image":image}
    return render(request, 'Upload2.html',{"show":False})
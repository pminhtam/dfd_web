from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
import json
# from django.core import serializers

from rest_framework.decorators import api_view, permission_classes
import numpy as np
import cv2

# @validate_decorator
# @api_view(["POST"])
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def detect_df(request):
    print(request)
    print(request.is_ajax())
    # print(request['image'])
    if request.method == "POST":
        # img = request.FILES['image'].read()
        img = request.POST.get('image')
        print(img)
        bytes = bytearray(img)
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        # print("numpyarray " ,numpyarray)
        # bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        # respone = json.dumps(image_boxs,cls=NumpyEncoder)
        # image_boxs_serialized = serializers.serialize('json', image_boxs)
        return JsonResponse({"aa":"khong co gi POST"},safe=False)
    else:
        return JsonResponse({"aa":"khong co gi GET"},safe=False)


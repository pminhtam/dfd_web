from django.urls import path
from . import views
from . import api

urlpatterns = [
    path('', views.detect_df, name='index'),
    path('detect/', views.detect_df, name='detect'),
    path('api/detect', api.detect_df, name='detect'),
    path('setting_detect/', views.setting_detect, name='setting detect'),
]
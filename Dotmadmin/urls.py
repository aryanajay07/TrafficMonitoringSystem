from django import views
from django.contrib import admin
from django.urls import path , include
from .views import *
urlpatterns = [
    path('', admin_login, name='dotm_login'),
    path("traffics/",traffics,name='traffics'),
    path("chart/",chart_display,name='chart'),
    path ("notice/",notice,name='notice'),
    path("home/",dotm_home,name='Home'),
]
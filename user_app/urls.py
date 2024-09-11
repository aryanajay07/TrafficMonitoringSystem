from . import views
from django.urls import path 
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("",views.welcome_page, name="welcome"),
    path('download-csv/', views.download_csv, name='download_csv'),
    path("Home/",views.home,name="home"),
    path('video/', views.video, name='video'),
    path("Records/",views.Records,name="Records"),
    path('api/records/', views.get_records, name='get_records'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

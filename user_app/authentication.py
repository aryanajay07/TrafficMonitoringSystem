from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from .models import Station

class MACAddressAuthBackend(ModelBackend):
    def authenticate(self, request, mac_address=None):
        User = get_user_model()
        try:
            station_user = Station.objects.get(mac_address=mac_address)
            user, _ = User.objects.get_or_create(username=f"station_{station_user.pk}")
            return user
        except Station.DoesNotExist:
            return None

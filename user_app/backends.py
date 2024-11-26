from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.models import User
from .models import Station

class MacAddressBackend(BaseBackend):
    def authenticate(self, request, mac_address=None):
        try:
            station = Station.objects.get(mac_address=mac_address)
            # Create or get a user for this station
            try:
                user = User.objects.get(username=f"station_{station.id}")
            except User.DoesNotExist:
                user = User.objects.create_user(
                    username=f"station_{station.id}",
                    password=None  # No password needed for MAC-based auth
                )
            return user
        except Station.DoesNotExist:
            return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None

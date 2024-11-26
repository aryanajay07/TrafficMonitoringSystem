from django.core.management.base import BaseCommand
from user_app.models import Station

class Command(BaseCommand):
    help = 'Creates a test station with the specified MAC address'

    def handle(self, *args, **kwargs):
        # Create a test station
        station = Station.objects.create(
            areacode=1,
            location='Test Location',
            mac_address='8c:aa:ce:51:67:e9'  # This is the MAC address used in views.py
        )
        self.stdout.write(self.style.SUCCESS(f'Successfully created test station with MAC address {station.mac_address}'))

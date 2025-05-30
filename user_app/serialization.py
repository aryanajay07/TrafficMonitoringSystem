from rest_framework import serializers
from .models import Record

class RecordSerializer(serializers.ModelSerializer):
    vehicle_image_url = serializers.SerializerMethodField()
    license_plate_image_url = serializers.SerializerMethodField()

    class Meta:
        model = Record
        fields = ['id', 'speed', 'date', 'count', 'licenseplate_no', 'vehicle_image_url', 'license_plate_image_url', 'stationID']  # Add other fields as necessary

    def get_vehicle_image_url(self, obj):
        request = self.context.get('request')
        if obj.vehicle_image:
            return request.build_absolute_uri(obj.vehicle_image.url)
        else:
            return request.build_absolute_uri('/speed_estimation/Test_video/jeep.jpg')  # Path to your default vehicle image

    def get_license_plate_image_url(self, obj):
        request = self.context.get('request')
        if obj.license_plate_image:
            return request.build_absolute_uri(obj.license_plate_image.url)
        else:
            return request.build_absolute_uri('speed_estimation/Test_video/jeep.jpg')  # Path to your default license plate image

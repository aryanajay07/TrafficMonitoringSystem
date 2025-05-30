from django.db import models

class Station(models.Model):
    areacode = models.PositiveIntegerField()
    location = models.CharField(max_length=80)
    mac_address = models.CharField(max_length=17)

    def __str__(self):
        return self.location

class Record(models.Model):
    stationID = models.ForeignKey('Station', on_delete=models.CASCADE)
    speed = models.IntegerField()
    date = models.DateField()
    count = models.IntegerField()
    licenseplate_no = models.CharField(max_length=50, null=True)
    vehicle_image = models.ImageField(upload_to='Vehicle_images/', default=None, null=True, blank=True)
    license_plate_image = models.ImageField(upload_to='License_plate_images/', default=None, null=True, blank=True)

    def __str__(self):
        return f"Record from {self.stationID}"

   



    


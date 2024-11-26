from speed_estimation.combinedmine import process_video
from django.shortcuts import render, redirect
from django.http import HttpResponse, StreamingHttpResponse
from .models import Record
import csv
from django.shortcuts import render
from django.contrib.auth import authenticate, login
import re, uuid
from rest_framework.response import Response
from django.views.decorators.csrf import ensure_csrf_cookie

@ensure_csrf_cookie
def welcome_page(request):
    try:
        if request.user.is_authenticated:
            return redirect('home')
        
        if request.method == 'POST':
            mac_address = '8c:aa:ce:51:67:e9'  # Fixed MAC address for testing
            user = authenticate(request, mac_address=mac_address)
            
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                return render(request, 'welcome_dashboard.html', {'error': 'Invalid MAC address. Consult DOTM'})
                
    except Exception as e:
        print(f"Authentication error: {e}")
        return render(request, 'welcome_dashboard.html', {'error': 'Authentication failed'})

    return render(request, 'welcome_dashboard.html')

@ensure_csrf_cookie
def home(request):
    try:
        if not request.user.is_authenticated:
            return redirect('welcome_page')
            
        Record_list = Record.objects.all()
        return render(request, 'base.html', {'Record_list': Record_list})

    except Exception as e:
        print(f"Home page error: {e}")
        return redirect('welcome_page')

def video(request):
    if not request.user.is_authenticated:
        return redirect('welcome_page')
    return StreamingHttpResponse(process_video(), content_type='multipart/x-mixed-replace; boundary=frame')

def Records(request):
    if not request.user.is_authenticated:
        return redirect('welcome_page')
    Record_list = Record.objects.all()
    context = {
        'Record_list': Record_list
    }
    return render(request, 'Records.html', context)

def download_csv(request):
    if not request.user.is_authenticated:
        return redirect('welcome_page')
    # Retrieve data from the database or any other source
    # records = Record.objects.all()  # Fetch records from the ViewRecord model
    license = request.GET.get('license')
    speed = request.GET.get('speed')
    date = request.GET.get('date')

    # Retrieve filtered records based on the search criteria
    filtered_records = Record.objects.all()  # Fetch all records by default

    if license:
        filtered_records = filtered_records.filter(licenseplate_no__icontains=license)

    if speed:
        filtered_records = filtered_records.filter(speed__icontains=speed)
	
    if date:
        filtered_records = filtered_records.filter(date__icontains=date)

    # Create a response object with CSV content
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="view_records.csv"'

    # Create a CSV writer and write the header row
    writer = csv.writer(response)
    writer.writerow(['SN', 'License Plate No', 'Speed', 'Date', 'ID', 'Count'])

    # Write the data rows
    for record in filtered_records:
        writer.writerow([
            record.pk,
            record.liscenseplate_no,
            record.speed,
            record.date,
            record.count
        ])

    return response

from django.http import JsonResponse
from rest_framework.decorators import api_view
from .serialization import RecordSerializer

@api_view(['GET'])
def get_records(request):
    if not request.user.is_authenticated:
        return redirect('welcome_page')
    records = Record.objects.all()
    serializer = RecordSerializer(records, many=True, context={'request': request})
    return Response(serializer.data)

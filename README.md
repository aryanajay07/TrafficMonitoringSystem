# 🚗 Smart Traffic Monitoring System

## 📝 Project Overview
A cutting-edge traffic monitoring solution that combines computer vision and deep learning to enhance road safety and traffic management. The system automatically detects vehicles, measures their speed, and recognizes license plates in real-time, providing a comprehensive solution for traffic enforcement and monitoring.

## 🎯 Key Features

### 🏃 Speed Detection
- Real-time vehicle speed measurement using advanced computer vision
- Multi-vehicle tracking with unique IDs
- Speed calculation using perspective transformation
- Configurable speed thresholds for different zones

### 📸 License Plate Recognition
- Automatic license plate detection and extraction
- High-accuracy OCR using EasyOCR
- Support for Nepali and English license plates
- Real-time plate number extraction and logging

### 🎥 Video Processing
- Support for both real-time video feeds and recorded footage
- Frame-by-frame analysis with object persistence
- Multi-threading for improved performance
- Efficient memory management for continuous operation

### 📊 Data Management
- Automatic logging of speed violations
- License plate image capture and storage
- Searchable database of violations
- Export functionality for reports and statistics

## 🛠️ Technical Architecture

### Core Components
1. **Vehicle Detection**
   - YOLOv8-based object detection
   - Custom-trained model for vehicle classification
   - Real-time object tracking with ByteTrack

2. **Speed Estimation**
   - Perspective transformation for accurate measurements
   - Time-based tracking system
   - Calibrated distance calculation

3. **License Plate Recognition**
   - Custom YOLO model for plate detection
   - EasyOCR integration for text extraction
   - Bilingual support (Nepali/English)

4. **Web Interface**
   - Django-based dashboard
   - Real-time monitoring interface
   - User authentication and role management
   - Violation review and management system

## 💻 Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- PostgreSQL database
- OpenCV dependencies

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-traffic-monitoring.git
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download model files (contact maintainer for access):
   - Place in `models/` directory:
     * 🔍 `Number_plate_recognize_last .pt`
     * 🚙 `Vehicle_Detection.pt`
     * 📋 `coco.txt`

5. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

6. Initialize database:
   ```bash
   python manage.py migrate
   ```

## 🚀 Usage

### Starting the System
1. Start the Django server:
   ```bash
   python manage.py runserver
   ```

2. Access the dashboard:
   - Open `http://localhost:8000` in your browser
   - Login with admin credentials

### Monitoring Setup
1. Configure monitoring zones
2. Set speed thresholds
3. Connect video source
4. Start monitoring

### Violation Management
1. Review detected violations
2. Export violation reports
3. Manage detected plates
4. Generate statistics

## 📊 Performance Metrics
- Vehicle Detection Accuracy: >95%
- License Plate Recognition Accuracy: >90%
- Speed Estimation Accuracy: ±3 km/h
- Processing Speed: 25-30 FPS (with GPU)

## 🔒 Security Features
- Encrypted data storage
- Role-based access control
- Audit logging
- Secure API endpoints

## 🌟 Use Cases

### 👮 Law Enforcement
- Automated speed violation detection
- Evidence collection and management
- Real-time alert system
- Historical data analysis

### 🚦 Traffic Management
- Traffic flow analysis
- Peak hour monitoring
- Accident prevention
- Infrastructure planning

### 🏢 Private Facilities
- Parking lot management
- Vehicle access control
- Security monitoring
- Fleet management

## 🔄 Future Enhancements
- [ ] AI-powered incident detection
- [ ] Mobile application development
- [ ] Cloud-based deployment option
- [ ] Integration with traffic signal systems
- [ ] Advanced analytics dashboard


## 🙏 Acknowledgments
- YOLOv8 team for object detection
- EasyOCR team for OCR capabilities
- ByteTrack for object tracking
- Django community for web framework
- OpenCV community for computer vision tools

---
Made with ❤️ for safer roads

import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                           QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import time

class DrowsinessDetector:
    def __init__(self):
        # Load the pre-trained Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Constants for drowsiness detection
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.25
        self.EYE_ASPECT_RATIO_CONSEC_FRAMES = 30  # ~2 seconds at 15 FPS
        
        # Initialize counters
        self.frame_counter = 0
        self.last_eye_check_time = time.time()

    def calculate_ear(self, eye_points):
        """Calculate the eye aspect ratio"""
        # Simplified EAR calculation using the height/width ratio of eye region
        height = eye_points[3] - eye_points[1]
        width = eye_points[2] - eye_points[0]
        ear = height / width if width != 0 else 0
        return ear

    def detect_drowsiness(self, frame):
        """
        Detect drowsiness in the given frame
        Returns: (processed_frame, is_drowsy)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        is_drowsy = False
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:  # We need at least two eyes for detection
                eyes_open = False
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    eye_points = [ex, ey, ex+ew, ey+eh]
                    ear = self.calculate_ear(eye_points)
                    
                    if ear > self.EYE_ASPECT_RATIO_THRESHOLD:
                        eyes_open = True
                        break
                
                if not eyes_open:
                    self.frame_counter += 1
                else:
                    self.frame_counter = 0
                
                # Check if eyes have been closed for too long
                if self.frame_counter >= self.EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    is_drowsy = True
            
            else:  # Reset counter if we can't detect eyes
                self.frame_counter = 0
        
        return frame, is_drowsy

class DrowsinessDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Initialize video capture and drowsiness detector
        self.cap = None
        self.detector = DrowsinessDetector()
        
        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Drowsiness detection state
        self.is_running = False
        self.warning_shown = False

    def initUI(self):
        """Initialize the GUI components"""
        self.setWindowTitle('Drowsiness Detection System')
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        
        # Create video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Create status label
        self.status_label = QLabel('Status: Not Running')
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create Start button
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)
        
        # Create Stop button
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        central_widget.setLayout(layout)

    def start_detection(self):
        """Start the drowsiness detection"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, 'Error', 'Could not access webcam!')
                return
        
        self.is_running = True
        self.timer.start(50)  # Update every 50ms (~20 FPS)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText('Status: Awake')

    def stop_detection(self):
        """Stop the drowsiness detection"""
        self.is_running = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText('Status: Not Running')
        self.video_label.clear()
        self.warning_shown = False

    def update_frame(self):
        """Update the video frame and check for drowsiness"""
        if not self.is_running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect drowsiness
        frame, is_drowsy = self.detector.detect_drowsiness(frame)
        
        # Update status and show warning if drowsy
        if is_drowsy:
            self.status_label.setText('Status: DROWSY!')
            if not self.warning_shown:
                self.warning_shown = True
                self.show_warning()
        else:
            self.status_label.setText('Status: Awake')
            self.warning_shown = False
        
        # Convert frame to QPixmap and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def show_warning(self):
        """Show warning message and initiate system shutdown/sleep"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Drowsiness Detected!")
        msg.setInformativeText("The system will initiate sleep mode in 10 seconds.")
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok)
        
        # Show message box and start shutdown timer
        msg.show()
        QTimer.singleShot(10000, self.initiate_sleep)

    def initiate_sleep(self):
        """Initiate system sleep/shutdown"""
        if sys.platform == "win32":
            os.system("shutdown /h")  # Hibernate on Windows
        else:
            os.system("systemctl suspend")  # Sleep on Linux
        self.stop_detection()

    def closeEvent(self, event):
        """Handle application closure"""
        self.stop_detection()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = DrowsinessDetectorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
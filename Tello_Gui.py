import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys
import argparse
import time
from djitellopy import Tello
import multiprocessing as mp
import math

# Import necessary YOLOv7 modules
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

print("\nTello Video Stream Program with Object Detection and Advanced Target Following\n")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    return opt

class VideoFeed(mp.Process):
    def __init__(self, frame_queue, command_queue, opt):
        super().__init__()
        self.frame_queue = frame_queue
        self.command_queue = command_queue
        self.opt = opt
        self.tello = None
        self.model = None
        self.device = None
        self.img_size = None
        self.names = None
        self.is_detecting = False
        self.is_tracking = False
        self.selected_target = None
        self.is_flying = False
        self.is_connected = False
        self.target_class = "person"
        self.desired_size_ratio = 0.3  # Target should occupy 30% of the frame
        self.size_tolerance = 0.05  # 5% tolerance

    def run(self):
        self.tello = Tello()
        
        # Initialize YOLOv7
        self.device = select_device(self.opt.device)
        print(f"Using device: {self.device}")
        self.model = attempt_load(self.opt.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(self.opt.img_size, s=self.stride)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        while True:
            if not self.command_queue.empty():
                command = self.command_queue.get()
                if command == "STOP":
                    break
                elif command == "CONNECT":
                    self.connect_to_drone()
                elif command == "DISCONNECT":
                    self.disconnect_from_drone()
                elif command == "TAKEOFF" and self.is_connected:
                    self.takeoff()
                elif command == "LAND" and self.is_connected:
                    self.land()
                elif command == "DETECT" and self.is_connected:
                    self.is_detecting = not self.is_detecting
                elif command == "TRACK" and self.is_connected:
                    self.is_tracking = not self.is_tracking
                elif command.startswith("SELECT_TARGET") and self.is_connected:
                    _, x, y = command.split()
                    self.select_target(int(x), int(y))
                elif command == "GET_TELEMETRY" and self.is_connected:
                    self.send_telemetry()

            if self.is_connected:
                frame = self.tello.get_frame_read().frame
                if self.is_detecting or self.is_tracking:
                    frame, detections = self.detect_objects(frame)
                    if self.is_tracking and self.selected_target:
                        self.track_selected_target(detections)
                
                self.frame_queue.put(frame)
            else:
                time.sleep(0.1)  # Sleep to prevent busy waiting when not connected

        if self.is_connected:
            self.tello.streamoff()
            self.tello.end()

    def connect_to_drone(self):
        if not self.is_connected:
            try:
                self.tello.connect()
                self.tello.streamon()
                self.is_connected = True
                print("Successfully connected to Tello drone.")
            except Exception as e:
                print(f"Failed to connect to Tello drone: {str(e)}")
                self.is_connected = False

    def disconnect_from_drone(self):
        if self.is_connected:
            try:
                self.tello.streamoff()
                self.tello.end()
                self.is_connected = False
                print("Disconnected from Tello drone.")
            except Exception as e:
                print(f"Error during disconnection: {str(e)}")

    def takeoff(self):
        if self.is_connected and not self.is_flying:
            try:
                print("Taking off...")
                self.tello.takeoff()
                self.is_flying = True
                print("Takeoff successful")

                print("Monitoring acceleration for 3 seconds...")
                start_time = time.time()
                while time.time() - start_time < 6:
                    accel_x = self.tello.get_acceleration_x()
                    accel_y = self.tello.get_acceleration_y()
                    accel_z = self.tello.get_acceleration_z()
                    
                    speed_x = self.tello.get_speed_x()
                    speed_y = self.tello.get_speed_y()
                    speed_z = self.tello.get_speed_z()
                    
                    print(f"Time: {time.time() - start_time:.2f}s")
                    print(f"Acceleration: x={accel_x:.2f}, y={accel_y:.2f}, z={accel_z:.2f} cm/s²")
                    print(f"Speed: x={speed_x:.2f}, y={speed_y:.2f}, z={speed_z:.2f} cm/s")
                    print("---")
                    
                    time.sleep(0.1)  # Small delay to prevent overwhelming the drone with requests

                # After monitoring, capture final values
                final_accel_x = self.tello.get_acceleration_x()
                final_accel_y = self.tello.get_acceleration_y()
                final_accel_z = self.tello.get_acceleration_z()

                final_speed_x = self.tello.get_speed_x()
                final_speed_y = self.tello.get_speed_y()
                final_speed_z = self.tello.get_speed_z()

                print("\nFinal readings after stabilization:")
                print(f"Acceleration: x={final_accel_x:.2f}, y={final_accel_y:.2f}, z={final_accel_z:.2f} cm/s²")
                print(f"Speed: x={final_speed_x:.2f}, y={final_speed_y:.2f}, z={final_speed_z:.2f} cm/s")

                # Check if the drone is hovering stably
                accel_threshold = 10  # cm/s², adjust as needed
                speed_threshold = 10  # cm/s, adjust as needed

                is_stable = (
                    abs(final_accel_x) < accel_threshold and
                    abs(final_accel_y) < accel_threshold and
                    abs(final_accel_z) < accel_threshold and
                    abs(final_speed_x) < speed_threshold and
                    abs(final_speed_y) < speed_threshold and
                    abs(final_speed_z) < speed_threshold
                )

                if is_stable:
                    print("Drone is hovering stably.")
                else:
                    print("Warning: Drone may not be hovering stably. Please check.")

                # Set the drone's speed to zero
                self.tello.send_rc_control(0, 0, 0, 0)
                print("Speed reset to zero.")

            except Exception as e:
                print(f"Takeoff failed: {str(e)}")
                self.is_flying = False
        else:
            print("Cannot takeoff: Drone is not connected or is already flying.")

    def land(self):
        if self.is_connected and self.is_flying:
            try:
                print("Landing...")
                self.tello.land()
                self.is_flying = False
                print("Landing successful")
            except Exception as e:
                print(f"Landing failed: {str(e)}")

    def detect_objects(self, img0):
        img = self.preprocess_image(img0)
        
        with torch.no_grad():
            pred = self.model(img)[0]

        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=None, agnostic=False)

        detections = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    self.plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=3)
                    detections.append((xyxy, self.names[int(cls)]))

        return img0, detections

    def preprocess_image(self, img0):
        img = cv2.resize(img0, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    @staticmethod
    def plot_one_box(x, img, color=(0, 255, 0), label=None, line_thickness=3):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def select_target(self, x, y):
        frame = self.tello.get_frame_read().frame
        _, detections = self.detect_objects(frame)
        
        for box, cls in detections:
            if cls == self.target_class:
                x1, y1, x2, y2 = map(int, box)
                if x1 < x < x2 and y1 < y < y2:
                    self.selected_target = {
                        'bbox': box,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'size': (x2 - x1) * (y2 - y1),
                        'size_ratio': ((x2 - x1) * (y2 - y1)) / (self.img_size * self.img_size)
                    }
                    print(f"Selected {self.target_class} at {box}")
                    return
        
        print(f"No {self.target_class} selected. Click on a detected {self.target_class} to start tracking.")

    def track_selected_target(self, detections):
        if not self.selected_target:
            return

        best_iou = 0
        best_detection = None

        for box, cls in detections:
            if cls == self.target_class:
                iou = self.calculate_iou(self.selected_target['bbox'], box)
                if iou > best_iou:
                    best_iou = iou
                    best_detection = box

        if best_detection:
            x1, y1, x2, y2 = map(int, best_detection)
            self.selected_target = {
                'bbox': best_detection,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'size': (x2 - x1) * (y2 - y1),
                'size_ratio': ((x2 - x1) * (y2 - y1)) / (self.img_size * self.img_size)
            }
            self.control_drone()
        else:
            self.tello.send_rc_control(0, 0, 0, 0)  # Stop if target lost

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection / float(area1 + area2 - intersection)
        return iou

    def control_drone(self):
        if not self.selected_target:
            return

        center_x, center_y = self.selected_target['center']
        frame_center_x, frame_center_y = self.img_size // 2, self.img_size // 2

        # Calculate horizontal and vertical movements
        yaw = (center_x - frame_center_x) / (self.img_size // 2)  # Normalized to [-1, 1]
        vertical = (frame_center_y - center_y) / (self.img_size // 2)  # Normalized to [-1, 1]

        # Calculate forward/backward movement based on size ratio
        size_diff = self.selected_target['size_ratio'] - self.desired_size_ratio
        forward = -size_diff / self.desired_size_ratio  # Normalized roughly to [-1, 1]

        # Scale movements (adjust these values as needed fast correction speeds can cause drone to crash)
        yaw_speed = int(yaw * 50)
        vertical_speed = int(vertical * 50)
        forward_speed = int(forward * 50)

        # Clamp speeds to acceptable range (-100 to 100)
        yaw_speed = max(-100, min(100, yaw_speed))
        vertical_speed = max(-100, min(100, vertical_speed))
        forward_speed = max(-100, min(100, forward_speed))

        # Only move if outside the tolerance zone
        if (abs(self.selected_target['size_ratio'] - self.desired_size_ratio) > self.size_tolerance or
            abs(center_x - frame_center_x) > self.img_size * 0.1 or
            abs(center_y - frame_center_y) > self.img_size * 0.1):
            self.tello.send_rc_control(0, forward_speed, vertical_speed, yaw_speed)
        else:
            self.tello.send_rc_control(0, 0, 0, 0)  # Hover if within tolerance

        # Print debug information
        print(f"Target center: ({center_x}, {center_y}), Size ratio: {self.selected_target['size_ratio']:.2f}")
        print(f"Control speeds - Yaw: {yaw_speed}, Vertical: {vertical_speed}, Forward: {forward_speed}")

    def send_telemetry(self):
        altitude = self.tello.get_height()
        speed = math.sqrt(self.tello.get_speed_x()**2 + self.tello.get_speed_y()**2 + self.tello.get_speed_z()**2)
        battery = self.tello.get_battery()
        temperature = self.tello.get_temperature()
        
        telemetry_data = {
            "altitude": altitude,
            "speed": speed,
            "battery": battery,
            "temperature": temperature
        }
        
        self.frame_queue.put(("TELEMETRY", telemetry_data))

class MainWindow(QMainWindow):
    def __init__(self, frame_queue, command_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.command_queue = command_queue
        self.setWindowTitle("Tello Drone Controller")
        self.setGeometry(100, 100, 800, 700)  # Increased height to accommodate new elements

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.video_label = ClickableLabel()
        self.layout.addWidget(self.video_label)

        # Telemetry display
        self.telemetry_layout = QHBoxLayout()
        self.layout.addLayout(self.telemetry_layout)

        self.altitude_label = QLabel("Altitude: N/A")
        self.speed_label = QLabel("Speed: N/A")
        self.battery_label = QLabel("Battery: N/A")
        self.temperature_label = QLabel("Temperature: N/A")

        self.telemetry_layout.addWidget(self.altitude_label)
        self.telemetry_layout.addWidget(self.speed_label)
        self.telemetry_layout.addWidget(self.battery_label)
        self.telemetry_layout.addWidget(self.temperature_label)

        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.connect_button = QPushButton("Connect to Drone")
        self.takeoff_button = QPushButton("Takeoff")
        self.land_button = QPushButton("Land")
        self.detect_button = QPushButton("Start Detection")
        self.track_button = QPushButton("Enable Tracking")

        self.button_layout.addWidget(self.connect_button)
        self.button_layout.addWidget(self.takeoff_button)
        self.button_layout.addWidget(self.land_button)
        self.button_layout.addWidget(self.detect_button)
        self.button_layout.addWidget(self.track_button)

        self.connect_button.clicked.connect(self.toggle_connection)
        self.takeoff_button.clicked.connect(self.takeoff)
        self.land_button.clicked.connect(self.land)
        self.detect_button.clicked.connect(self.toggle_detection)
        self.track_button.clicked.connect(self.toggle_tracking)

        self.video_label.clicked.connect(self.on_video_click)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frame)
        self.update_timer.start(33)  # Update at approximately 30 FPS

        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self.update_telemetry)
        self.telemetry_timer.start(1000)  # Update telemetry every second

        # Initially disable buttons
        self.takeoff_button.setEnabled(False)
        self.land_button.setEnabled(False)
        self.detect_button.setEnabled(False)
        self.track_button.setEnabled(False)

    def toggle_connection(self):
        if self.connect_button.text() == "Connect to Drone":
            self.command_queue.put("CONNECT")
            self.connect_button.setText("Disconnect Drone")
            self.takeoff_button.setEnabled(True)
            self.detect_button.setEnabled(True)
        else:
            self.command_queue.put("DISCONNECT")
            self.connect_button.setText("Connect to Drone")
            self.takeoff_button.setEnabled(False)
            self.land_button.setEnabled(False)
            self.detect_button.setEnabled(False)
            self.track_button.setEnabled(False)
            # Reset button texts
            self.detect_button.setText("Start Detection")
            self.track_button.setText("Enable Tracking")

    def takeoff(self):
        self.command_queue.put("TAKEOFF")
        self.takeoff_button.setEnabled(False)
        self.land_button.setEnabled(True)

    def land(self):
        self.command_queue.put("LAND")
        self.land_button.setEnabled(False)
        self.takeoff_button.setEnabled(True)

    def toggle_detection(self):
        self.command_queue.put("DETECT")
        if self.detect_button.text() == "Start Detection":
            self.detect_button.setText("Stop Detection")
            self.track_button.setEnabled(True)
        else:
            self.detect_button.setText("Start Detection")
            self.track_button.setEnabled(False)
            self.track_button.setText("Enable Tracking")

    def toggle_tracking(self):
        self.command_queue.put("TRACK")
        if self.track_button.text() == "Enable Tracking":
            self.track_button.setText("Disable Tracking")
        else:
            self.track_button.setText("Enable Tracking")

    def update_frame(self):
        if not self.frame_queue.empty():
            data = self.frame_queue.get()
            if isinstance(data, tuple) and data[0] == "TELEMETRY":
                telemetry = data[1]
                self.update_telemetry_display(
                    telemetry["altitude"],
                    telemetry["speed"],
                    telemetry["battery"],
                    telemetry["temperature"]
                )
            else:
                frame = data
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

    def update_telemetry(self):
        # Request telemetry data from the drone
        self.command_queue.put("GET_TELEMETRY")

    def update_telemetry_display(self, altitude, speed, battery, temperature):
        self.altitude_label.setText(f"Altitude: {altitude:.2f} cm")
        self.speed_label.setText(f"Speed: {speed:.2f} cm/s")
        self.battery_label.setText(f"Battery: {battery}%")
        self.temperature_label.setText(f"Temperature: {temperature:.1f}°C")

    def on_video_click(self, point):
        self.command_queue.put(f"SELECT_TARGET {point.x()} {point.y()}")

    def closeEvent(self, event):
        self.command_queue.put("STOP")
        event.accept()

class ClickableLabel(QLabel):
    clicked = pyqtSignal(QPoint)

    def mousePressEvent(self, event):
        self.clicked.emit(event.pos())

if __name__ == '__main__':
    opt = parse_opt()
    
    frame_queue = mp.Queue(maxsize=2)
    command_queue = mp.Queue()

    video_feed = VideoFeed(frame_queue, command_queue, opt)
    video_feed.start()

    app = QApplication(sys.argv)
    window = MainWindow(frame_queue, command_queue)
    window.show()

    sys.exit(app.exec_())

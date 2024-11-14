import os
import cv2
import numpy as np
import time
import requests
from requests.auth import HTTPDigestAuth
from ultralytics import YOLO
from configparser import ConfigParser
import logging
import base64
import json
from datetime import datetime
import pytz

# Camera and authentication details
snapshot_url1 = 'http://172.11.20.212/ISAPI/Streaming/channels/1/picture' #VIDS Cam Lane 2
snapshot_url2 = 'http://172.11.20.210/ISAPI/Streaming/channels/1/picture' #VIDS CamLane 1
username = 'admin'
password = 'Ador2024'

# Directory to save snapshots
save_directory = 'snapshots'
violation_directory = 'violations'
os.makedirs(save_directory, exist_ok=True)
os.makedirs(violation_directory, exist_ok=True)

# Logging setup
timezone = pytz.timezone('Asia/Kolkata')
c_date = datetime.now(timezone)
date1 = c_date.strftime("%Y_%m_%d")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("my_logger1")
handler = logging.FileHandler(f'vids_puri_rhs_{date1}.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load model configurations
config_object = ConfigParser()
config_object.read("/root/apps/config_puri_rhs.ini")
parameters = config_object["project_parameters"]
# Load YOLO models
vehicle_model = YOLO(parameters['model'])

# Print and log all class names inside the vehicle model
logger.info("Class names inside the vehicle model:")
for idx, class_name in enumerate(vehicle_model.names):
    logger.info(f"Class index: {idx}, Class name: {class_name}")


# API calling function
def send_frame_to_api(frame, cropped_frame, cam_id, timestamp, d, bb, label):
    print("here is your api")
    logger.info(f'{timestamp}_Inside API')
    print(f'{timestamp}_Inside API')
    api_url = parameters['apiURL']
    
    try:
        # Convert the buffer to a base64 string for the full image
        _, buffer_full = cv2.imencode('.jpg', frame)
        image_base64_full = base64.b64encode(buffer_full).decode('utf-8')

        # Convert the buffer to a base64 string for the cut image (cropped violation image)
        _, buffer_cut = cv2.imencode('.jpg', cropped_frame)
        image_base64_cut = base64.b64encode(buffer_cut).decode('utf-8')

        payload = json.dumps({
            "EventLog": {
                "LPUID": "48:b0:2d:60:10:c3",
                "CameraID": cam_id,
                "Timestamp": timestamp,
                "Direction": d,
                "VehicleCatagory": "na",
                "ANPR": "na",
                "Coordinates": {
                    "x1": bb[0],
                    "y1": bb[1],
                    "x2": bb[2],
                    "y2": bb[3]
                },
                "Incidents": [
                    {
                        "x1": bb[0],
                        "y1": bb[1],
                        "x2": bb[2],
                        "y2": bb[3],
                        "label": label
                    }
                ],
                "Base64ImageFull": image_base64_full,
                "Base64ImageCut": image_base64_cut
            }
        })
        headers = {
            'Accept': '*/*',
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", api_url, headers=headers, data=payload)
        logger.info(f'response : {response.text}')

    except Exception as e:
        logger.error(f'Error: API Function : {e}')


# Save violation image and call API
# Save violation image and call API
def save_violation_image_and_call_api(original_frame, frame, bbox, violation_type, timestamp, camera_id):
    # Crop the vehicle's image using the bounding box directly from the original frame
    x1, y1, x2, y2 = bbox
    cropped_image = original_frame[y1:y2, x1:x2]  # Crop from the original frame to avoid drawings

    # Create the violation image filename
    violation_filename = f"{violation_type}_{camera_id}_{timestamp}.jpg"
    violation_path = os.path.join(violation_directory, violation_filename)

    # Save the cropped image (without any boundary or label)
    cv2.imwrite(violation_path, cropped_image)
    logger.info(f"Violation image saved: {violation_path}")

    # Send the full frame and cropped image to the API
    send_frame_to_api(frame, cropped_image, camera_id, timestamp, 'Entry', bbox, violation_type)




# Check for violations within vehicle bounding box
# Check for violations within vehicle bounding box
def check_violations(boxes, c_ids, bbox, original_frame, timestamp, camera_id):
    # Define violation classes (indices for 'mobile_calling', 'mobile_watching', 'no_helmet', 'no_seatbelt')
    violation_classes = [1, 2, 3, 4]
    vehicle_bbox = None
    detected_violation = False
    violation_type = ""
    rider_count = 0

    # Iterate over detected objects
    for box2, cls2 in zip(boxes, c_ids):
        # Calculate the coordinates of the object
        x1, y1, x2, y2 = int(box2[0]), int(box2[1]), int(box2[2]), int(box2[3])

        # Check if at least part of the object is within the vehicle bounding box
        if bbox[0] <= x2 and bbox[2] >= x1 and bbox[1] <= y2 and bbox[3] >= y1:
            # Count as a rider if it's either 'helmet' or 'no_helmet'
            if cls2 in [0, 3]:  # 'helmet' and 'no_helmet'
                rider_count += 1

            # If the object is a violation class
            if cls2 in violation_classes:
                detected_violation = True
                violation_type = vehicle_model.names[int(cls2)]

    # Check for triple riding violation
    if rider_count >= 3:
        detected_violation = True
        violation_type = 'triple_riding'

    # If a violation is detected within this vehicle
    if detected_violation:
        vehicle_bbox = bbox
        logger.info(f'Violation detected within vehicle bounding box: {violation_type}')
        print('-----------------')  # Console print for violation detection
        save_violation_image_and_call_api(original_frame, original_frame, vehicle_bbox, violation_type, timestamp, camera_id)


# Function to capture and process images from a camera
def capture_from_camera(snapshot_url, camera_id):
    try:
        logger.info(f"Capturing image from camera {camera_id}.")
        # Capture snapshot from camera with a timeout of 10 seconds
        response = requests.get(snapshot_url, auth=HTTPDigestAuth(username, password), timeout=10)

        if response.status_code == 200:
            # Get the current timestamp in the desired format
            timestamp = datetime.now(timezone).strftime("%Y_%m_%d_%H_%M_%S")
            filename = os.path.join(save_directory, f'snapshot_{camera_id}_{timestamp}.jpg')
            with open(filename, 'wb') as f:
                f.write(response.content)
            logger.info(f"Snapshot saved: {filename}")

            # Process the captured image
            original_frame = cv2.imread(filename)  # Read the original frame
            if original_frame is None:
                logger.error(f"Error: Unable to read the image {filename}")
                return

            logger.info("Running object detection on the captured image.")
            result = vehicle_model(original_frame)[0]
            frame = result.plot()

            # Check if any vehicles were detected
            if len(result.boxes) > 0:
                bboxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls

                # Log detected classes
                logger.info(f"Detected classes: {class_ids}")

                # Iterate over detected objects
                for cls, bb in zip(class_ids, bboxes):
                    bb = [int(x) for x in bb]
                    x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]

                    # Check if the detected object is a vehicle
                    if cls == 6:  # Class index for 'vehicle'
                        # Log bounding box coordinates
                        logger.info(f'Vehicle Bounding Box Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}')

                        # Check for any violations within this vehicle's bounding box
                        check_violations(bboxes, class_ids, bb, original_frame, timestamp, camera_id)  # Pass the original frame
            else:
                logger.info(f"No vehicles detected in image: {filename}")
                # os.remove(filename)  # Delete the image if no vehicles are detected
                logger.info(f"Deleted image without vehicles: {filename}")

        else:
            logger.error(f"Error: Unable to capture snapshot from camera {camera_id}. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error: An error occurred while capturing snapshot from camera {camera_id}: {e}")

# Function to capture and process images from both cameras
def capture_and_process():
    try:
        while True:
            # Capture and process images from both cameras
            capture_from_camera(snapshot_url1, 'camera1')
            capture_from_camera(snapshot_url2, 'camera2')

            # Wait for 1 second before capturing the next snapshots
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping the snapshot capture and processing.")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_and_process()

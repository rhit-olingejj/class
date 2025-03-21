import os.path

import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# Tweak this value depending on how confident we want to filter by
CONFIDENCE_THRESHOLD = 0.7


# Allow user to select a file from the PC
def select_file_dialog():
    image_extensions = ['.jpg', '.jpeg', '.png']
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm']
    file_type = None
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a file")
    if file_path:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in image_extensions:
            file_type = "Image"
        elif file_extension in video_extensions:
            file_type = "Video"
        else:
            file_type = None
    return file_path, file_type


# Create output folder into the current directory under the file name's name
def make_output_folder(file_path):
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, file_name_without_extension)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


# Handles Image detection using YOLO
def yolo_image_detection(model, file_path):
    # Create a new folder to place the detected images in
     output_folder_name = make_output_folder(file_path)

     # Read the selected image
     image = cv2.imread(file_path)

     # Run YOLO detection
     results = model(image)[0]

     # Loop through detections
     for i, (box, cls, conf) in enumerate(zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)):
         class_name = results.names[int(cls)]
         confidence = float(conf)

         # Only process vehicles
         if class_name in ["car", "truck", "bus"]:
             # Ignore all detected classes under our defined CONFIDENCE_THRESHOLD
             if confidence > CONFIDENCE_THRESHOLD:
                 x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
                 vehicle_crop = image[y1:y2, x1:x2]  # Crop vehicle

                 # Save cropped vehicle image
                 output_file_name = os.path.join(output_folder_name, f"vehicle_{i}.jpg")
                 cv2.imwrite(output_file_name, vehicle_crop)

                 # Draw the bounding box on the image
                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)

     # Show the full image with bounding boxes
     cv2.imshow("Image with Bounding Boxes", image)
     cv2.waitKey(0)
     cv2.destroyAllWindows()


# Handles Video detection using YOLO
def yolo_video_detection(model, file_path):
    # Create a new folder to place the detected images in
    output_folder_name = make_output_folder(file_path)

    # Open the video
    video_capture = cv2.VideoCapture(file_path)

    # Double check that the video can be opened
    if not video_capture.isOpened():
        print("Error: Could not open video")
        return

    # Grab the FPS for the video to determine how long 1 second is
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  # Break if no frame is read

        frame_count += 1

        # Skip frames until we reach the 1-second mark (We only want to process every one second)
        if frame_count % int(fps) == 0:
            # Run YOLO detection on the current frame
            results = model(frame)[0]

            # Loop through detections
            for i, (box, cls, conf) in enumerate(zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)):
                class_name = results.names[int(cls)]
                confidence = float(conf)

                # Only process vehicles (car, truck, bus, motorcycle)
                if class_name in ["car", "truck", "bus"]:
                    # Ignore all detected classes under our defined CONFIDENCE_THRESHOLD
                    if confidence > CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
                        vehicle_crop = frame[y1:y2, x1:x2]  # Crop vehicle

                        # Save vehicle cropped image (Named with frame count and detection index)
                        output_file_name = os.path.join(output_folder_name, f"vehicle_{frame_count // int(fps)}_{i}.jpg")
                        cv2.imwrite(output_file_name, vehicle_crop)

                        # Draw bounding boxes around the vehicles
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the frame with bounding boxes
            cv2.imshow("Frame", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Load YOLOv8 model
    model = YOLO("yolov8s.pt")

    # Ask user to choose a file and process it accordingly
    file_path, file_type = select_file_dialog()
    if file_path:
        if file_type == "Image":
            yolo_image_detection(model, file_path)
        elif file_type == "Video":
            yolo_video_detection(model, file_path)
        else:
            print("File type not supported")


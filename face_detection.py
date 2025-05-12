"""
Face Detection Project using OpenCV and Python
This script provides functionality for detecting faces in both images and webcam feeds.
Author: Melisa Sever
Date: May 12, 2025
"""

import cv2
import sys
import numpy as np
import os
from datetime import datetime

class FaceDetector:
    def __init__(self):
        # Load the pre-trained face detector (Haar Cascade Classifier)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Also load eye detector for additional features
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Create directory for saved images if it doesn't exist
        self.output_dir = 'detected_faces'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def detect_faces_in_image(self, image_path):
        """Detect faces in a single image file"""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle for face
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract region of interest (ROI) for eyes detection
            roi_gray = gray_image[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            # Detect eyes in the face
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Display the result
        print(f"Found {len(faces)} faces!")
        cv2.imshow('Detected Faces', image)
        
        # Save the detected image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.output_dir}/detected_{timestamp}.jpg"
        cv2.imwrite(output_filename, image)
        print(f"Image saved as {output_filename}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def detect_faces_webcam(self):
        """Real-time face detection using webcam"""
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Face detection started. Press 'q' to quit, 's' to save current frame.")
        
        # Variables to track stats
        frame_count = 0
        total_faces = 0
        
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                # ROI for eyes
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Update stats
            frame_count += 1
            total_faces += len(faces)
            avg_faces = total_faces / frame_count if frame_count > 0 else 0
            
            # Display stats
            cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Avg faces per frame: {avg_faces:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the resulting frame
            cv2.imshow('Webcam Face Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Quit
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{self.output_dir}/webcam_{timestamp}.jpg"
                cv2.imwrite(output_filename, frame)
                print(f"Frame saved as {output_filename}")
        
        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()
    
    def detect_faces_advanced(self, use_webcam=True, image_path=None):
        """
        Advanced face detection using OpenCV DNN module
        Much better accuracy than Haar cascades
        """
        # Check if we have the DNN face detector models
        model_files = {
            "model": "models/res10_300x300_ssd_iter_140000.caffemodel",
            "config": "models/deploy.prototxt"
        }
        
        # Check if models directory exists
        if not os.path.exists("models"):
            os.makedirs("models")
            print("Created 'models' directory. Please download the model files:")
            print("1. res10_300x300_ssd_iter_140000.caffemodel")
            print("2. deploy.prototxt")
            print("and place them in the 'models' directory.")
            return
        
        # Check if model files exist
        for key, file_path in model_files.items():
            if not os.path.exists(file_path):
                print(f"Missing {key} file: {file_path}")
                print("Please download the model files as instructed.")
                return
        
        # Load the DNN face detector
        print("Loading DNN face detector model...")
        net = cv2.dnn.readNetFromCaffe(model_files["config"], model_files["model"])
        
        if use_webcam:
            self._process_webcam_advanced(net)
        else:
            if image_path:
                self._process_image_advanced(net, image_path)
            else:
                print("Error: No image path provided for image processing mode")
    
    def _process_image_advanced(self, net, image_path):
        """Process a single image with the DNN model"""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Get image dimensions
        (h, w) = image.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the network
        net.setInput(blob)
        detections = net.forward()
        
        # Process detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > 0.5:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw the bounding box and confidence
                text = f"{confidence * 100:.2f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow("DNN Face Detection", image)
        
        # Save the detected image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.output_dir}/dnn_detected_{timestamp}.jpg"
        cv2.imwrite(output_filename, image)
        print(f"Image saved as {output_filename}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _process_webcam_advanced(self, net):
        """Process webcam feed with the DNN model"""
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Advanced face detection started. Press 'q' to quit, 's' to save current frame.")
        
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Get frame dimensions
            (h, w) = frame.shape[:2]
            
            # Create a blob from the image
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            
            # Pass the blob through the network
            net.setInput(blob)
            detections = net.forward()
            
            # Process detections
            face_count = 0
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter weak detections
                if confidence > 0.5:
                    face_count += 1
                    
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Draw the bounding box and confidence
                    text = f"{confidence * 100:.2f}%"
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            # Display the resulting frame with face count
            cv2.putText(frame, f'Faces detected: {face_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Advanced Face Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Quit
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{self.output_dir}/dnn_webcam_{timestamp}.jpg"
                cv2.imwrite(output_filename, frame)
                print(f"Frame saved as {output_filename}")
        
        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()


def main():
    """Main function to parse arguments and run the selected detection mode"""
    detector = FaceDetector()
    
    if len(sys.argv) < 2:
        print("Face Detection System")
        print("Usage options:")
        print("1. Detect faces in an image:")
        print("   python face_detection.py --image path/to/image.jpg")
        print("2. Detect faces using webcam:")
        print("   python face_detection.py --webcam")
        print("3. Advanced face detection using DNN and webcam:")
        print("   python face_detection.py --advanced-webcam")
        print("4. Advanced face detection using DNN on an image:")
        print("   python face_detection.py --advanced-image path/to/image.jpg")
        
        # Default to webcam mode if no arguments
        print("\nNo arguments provided. Starting webcam detection mode...")
        detector.detect_faces_webcam()
        return
    
    # Parse arguments
    if sys.argv[1] == "--image" and len(sys.argv) > 2:
        detector.detect_faces_in_image(sys.argv[2])
    elif sys.argv[1] == "--webcam":
        detector.detect_faces_webcam()
    elif sys.argv[1] == "--advanced-webcam":
        detector.detect_faces_advanced(use_webcam=True)
    elif sys.argv[1] == "--advanced-image" and len(sys.argv) > 2:
        detector.detect_faces_advanced(use_webcam=False, image_path=sys.argv[2])
    else:
        print("Invalid arguments. Use --help for usage instructions.")


if __name__ == "__main__":
    main()
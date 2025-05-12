# Face Detection Project
A Python project for detecting faces in images and webcam feeds using OpenCV.

## Author
Melisa

## Features
- Face detection in static images
- Real-time face detection using webcam
- Advanced face detection using OpenCV's DNN module
- Eye detection within detected faces
- Saving detected images with timestamps
- Statistics tracking for webcam mode

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install the required packages:
```
pip install opencv-python numpy
```

## Usage
### Basic Usage
Run the script with no arguments to start webcam detection:
```
python face_detection.py
```

### Other Options
1. Detect faces in an image:
   ```
   python face_detection.py --image path/to/image.jpg
   ```

2. Detect faces using webcam:
   ```
   python face_detection.py --webcam
   ```

3. Advanced face detection using DNN and webcam:
   ```
   python face_detection.py --advanced-webcam
   ```

4. Advanced face detection using DNN on an image:
   ```
   python face_detection.py --advanced-image path/to/image.jpg
   ```

## Advanced DNN-based Detection
For better detection results, download these model files:
1. [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
2. [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)

Place them in a "models" directory in the same folder as the script.

## Controls
- Press 'q' to quit
- Press 's' to save the current frame (in webcam modes)

## Output
Detected faces are saved in the "detected_faces" directory with a timestamp in the filename.

## Privacy Note
This project includes a `.gitignore` file that prevents detected face images from being tracked in version control. If you're concerned about privacy, make sure to:
1. Use the provided `.gitignore` file
2. Don't manually commit any images from the "detected_faces" directory
3. Be cautious about where you share or store these images

## License
This project is for educational purposes only. Feel free to modify and use it for personal projects.

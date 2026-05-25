# 🧱 Lego Piece Scanner Prototype

This software helps you keep track of all your Lego pieces by scanning images, detecting the pieces, and providing a sortable, exportable list.

Currently, this acts as a **prototype** demonstrating the workflow using a standard object detection model (YOLOv8). It identifies general objects in the image and simulates assigning Lego part numbers and colors.

## Features
- Cross-platform Web UI built with Streamlit (works on Windows/Mac/Linux).
- Object detection using YOLOv8 to draw bounding boxes around pieces.
- Tabular display of detected pieces with summaries.
- Export results as CSV or JSON files.

## Installation

1. Make sure you have Python 3.8+ installed.
2. Clone this repository.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the scanner web interface, run the following command in your terminal:

```bash
streamlit run app.py
```

A browser window will open automatically at `http://localhost:8501`.

## How to Upgrade to Real Part Numbers (Training Guide)

This application is built to be easily upgraded. The generic YOLOv8 model (`yolov8n.pt`) doesn't know specific Lego pieces. To make it detect actual part numbers, you need to train a custom model:

1. **Gather Data:** Take pictures of your Lego pieces on a clean background.
   * **Pro-tip:** Instead of taking hundreds of pictures manually, you can take a 30-second video walking around your Lego pieces. Then, use the included utility script to extract the frames automatically:
     ```bash
     python extract_frames.py my_lego_video.mp4 --interval 10
     ```
     This will pull every 10th frame and save it as a high-quality image in the `dataset_frames` folder, ready for annotation!
2. **Annotate:** Use a tool like [Roboflow](https://roboflow.com/) or CVAT to upload your images, draw boxes around the pieces, and label them with their part number (e.g., `3001`).
3. **Train:** Export the dataset in YOLOv8 format and use the Ultralytics library to train your custom model:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   results = model.train(data='your_lego_dataset.yaml', epochs=100)
   ```
4. **Deploy:** Once training is done, you will get a file named `best.pt`. Copy this file into this project's folder.
5. **Update Code:** Open `detector.py` and change the line `model = YOLO('yolov8n.pt')` to `model = YOLO('best.pt')`. You can then remove the random placeholder logic, as your model will natively output the correct part numbers!

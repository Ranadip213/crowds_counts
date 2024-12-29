Person Detection and Counting Project

Overview

This project focuses on detecting and counting the number of people in images using the EfficientDet object detection model from TensorFlow Hub. It includes functionality for drawing bounding boxes around detected individuals and analyzing the results statistically.

Features

Object Detection:

Detect objects in images using the pre-trained EfficientDet model.

Identify and filter objects classified as "person".

Person Counting:

Count the number of people in each image based on detection results and confidence thresholds.

Visualization:

Overlay bounding boxes on images to highlight detected individuals.

Statistical Analysis:

Analyze and visualize metadata such as the distribution of people counts in a dataset.

Applications

Crowd Counting: Estimating the size of crowds in public spaces or events.

Smart Cities: Monitoring pedestrian traffic and optimizing urban planning.

Retail Analytics: Analyzing customer footfall in stores or malls.

Emergency Management: Identifying and managing crowded areas for safety.

Installation

Clone this repository:

[git clone https://github.com/your-username/person-detection.git
cd person-detection]

Install the required dependencies:

pip install -r requirements.txt

If using Google Colab:

Upload your images and metadata (e.g., labels.csv) to your Colab environment.

Mount your Google Drive if required.

Usage

Running the Project

Prepare Metadata:

Ensure you have a CSV file (e.g., labels.csv) with image IDs and true counts of people.

Update the META_FILE path in the script to point to your CSV file.

File Path Reconstruction:

The script reconstructs image file paths based on IDs in the metadata.

Object Detection:

Load the EfficientDet model:

detector = hub.load(MODEL_PATH)

Perform detection and count persons:

results = detect_objects(image_path, detector)
count = count_persons(image_path, detector, threshold=0.5)

Visualization:

Draw bounding boxes on an image:

image_with_bboxes = draw_bboxes(image_path, results, threshold=0.5)
image_with_bboxes.show()

Statistical Analysis:

Analyze and visualize the dataset:

stats = data.describe()
plt.hist(data['count'], bins=20)
plt.show()

Example

example_path = '/path/to/your/image.jpg'
results = detect_objects(example_path, detector)
draw_bboxes(example_path, results, threshold=0.5)

Directory Structure

project/
|-- frames/
|   |-- seq_000001.jpg
|   |-- seq_000002.jpg
|   `-- ...
|-- labels.csv
|-- script.py
|-- README.md
`-- requirements.txt

Requirements

Python 3.8+

TensorFlow 2.x

TensorFlow Hub

NumPy

Pandas

Matplotlib

tqdm

PIL

Limitations

False Positives: Non-human objects may sometimes be misclassified as people.

Model Accuracy: The pre-trained EfficientDet model may require fine-tuning for specific datasets.

File Path Issues: Ensure correct file paths for input images and metadata.

Contributing

Contributions are welcome! Please follow these steps:

Fork the repository.

Create a new branch for your feature or bug fix.

Commit your changes and submit a pull request.




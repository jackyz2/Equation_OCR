from ultralytics import YOLO

# Load the YOLOv8 model, you can choose between 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', or 'yolov8x'
model = YOLO('yolov8n.pt')  # 'n' stands for nano, change it according to your needs

# Train the model on the dataset
model.train(
    data='./dataset/data.yaml',  # Path to your data.yaml file inside the dataset folder
    epochs=1000,  # Number of training epochs
    batch=16,    # Batch size, you can adjust this based on your hardware
    imgsz=640,   # Image size, 640 is the default, can also be 416, 512, etc.
    project='train_results',  # Output directory for training results
    name='yolov8_equations'   # Name for the experiment
)

# Optionally, after training, you can evaluate the model on the test set
metrics = model.val(data='./dataset/data.yaml')  # Evaluate the model on the validation dataset

# To predict on new images or run inference, you can use the following:
results = model.predict(source='/Users/jackyzhang/ClassTranscribeOCR/dataset/test/images/image3_png.rf.0317c94d94dfefe6390fbfd69503f92f.jpg')  # Replace with the path to your test images

from ultralytics import YOLO


def main():
    # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    # model.train(data='coco128.yaml', epochs=100, imgsz=640)

    # Predict with the model
    results = model('https://ultralytics.com/images/bus.jpg', save=True)  # predict on an image


if __name__ == "__main__":
    main()

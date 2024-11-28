from ultralytics import YOLO

model = YOLO(model='yolo11n-obb.pt')

model.train(data='dataset/data.yaml', epochs=1)

model.export(format='onnx')
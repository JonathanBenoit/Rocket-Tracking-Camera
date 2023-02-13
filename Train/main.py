from ultralytics import YOLO
from roboflow import Roboflow
import os
import shutil
from dotenv import load_dotenv

def dowload_dataset(dataset_version):
    if(os.path.exists("Train/datasets/RockETS-Rockets-" + str(dataset_version))):
        print("Dataset already downloaded")
        return

    print("Downloading dataset...")
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')    
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("whowho").project("rockets-rockets")
    dataset = project.version(dataset_version).download("yolov8")
    shutil.move(dataset.location, "Train/datasets/" + dataset.location.split("\\")[-1])

def train(model, dataset_version):
    model.train(data="Train/datasets/RockETS-Rockets-" + str(dataset_version) + "/data.yaml", epochs=1, imgsz=640, workers=4)

def test(model):
    results = model.predict("Train/test/aarluk-III-LCE.jpg")
    printResults(results)

def printResults(results):
    if(results is None):
        return

    for result in results:
        boxes = result.boxes
        print(boxes)


if __name__ == "__main__":
    load_dotenv()
    dataset_version = 9
    model = YOLO("Train/runs/detect/train31/weights/best.pt")
    dowload_dataset(dataset_version)
    train(model, dataset_version)
    test(model)

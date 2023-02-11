from ultralytics import YOLO
from roboflow import Roboflow
import os
import shutil


def dowload_dataset(dataset_version):
    if(os.path.exists("datasets\RockETS-Rockets-" + str(dataset_version))):
        print("Dataset already downloaded")
        return

    print("Downloading dataset...")
    rf = Roboflow(api_key="SOPMIV7bORFJq4oRfvDm")
    project = rf.workspace("whowho").project("rockets-rockets")
    dataset = project.version(dataset_version).download("yolov8")
    shutil.move(dataset.location, "F:/Workspace/Ecole/Session 8/PFE/Rocket-Tracking-Camera/datasets/" + dataset.location.split("\\")[-1])

def train(model, dataset_version):
    model.train(data="datasets\RockETS-Rockets-"+ str(dataset_version) + "\data.yaml", epochs=20, imgsz=640, device=0, workers=4)
    results = model.val()
    print(results)

def test(model):
    results = model("Aarluk-III.jpg")
    printResults(results)

def printResults(results):
    if(results is None):
        return

    for result in results:
        if(result.boxes.xyxy):
            print(result.boxes.xyxy)

if __name__ == "__main__":
    dataset_version = 8
    model = YOLO("yolov8n.yaml")
    dowload_dataset(dataset_version)
    train(model, dataset_version)
    test(model)

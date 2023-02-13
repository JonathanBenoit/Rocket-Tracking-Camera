from ultralytics import YOLO
from roboflow import Roboflow
import os
import shutil
from PIL import Image
import numpy as np
import cv2


def dowload_dataset(dataset_version):
    if(os.path.exists("datasets/RockETS-Rockets-" + str(dataset_version))):
        print("Dataset already downloaded")
        return

    print("Downloading dataset...")
    rf = Roboflow(api_key="SOPMIV7bORFJq4oRfvDm")
    project = rf.workspace("whowho").project("rockets-rockets")
    dataset = project.version(dataset_version).download("yolov8")
    shutil.move(dataset.location, "C:/Users/RockETS/Documents/Justin/PFE/Rocket-Tracking-Camera/datasets/" + dataset.location.split("\\")[-1])

def train(model, dataset_version):
    model.train(data="C:/Users/RockETS/Documents/Justin/PFE/Rocket-Tracking-Camera/datasets/RockETS-Rockets-9/data.yaml", epochs=100, imgsz=640, workers=4)

def test(model):
    results = model("test.webp", save=True)
    printResults(results)

def printResults(results):
    if(results is None):
        return

    for result in results:
        boxes = result.boxes
        print(boxes)


if __name__ == "__main__":
    dataset_version = 9
    model = YOLO("runs\\detect\\train31\\weights\\best.pt")
    #dowload_dataset(dataset_version)
    #train(model, dataset_version)
    test(model)

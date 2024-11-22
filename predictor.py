from ultralytics import YOLO

bestModel = YOLO("C:/Users/Alejandro/Desktop/Trash/runs/classify/train4/weights/best.pt")

resultsPrediction = bestModel("C:/Users/Alejandro/Desktop/Trash/PRUEBAVALIDACIONCARDBOARD.jpg")

nombresDicc = resultsPrediction[0].names

probs = resultsPrediction[0].probs

print(probs)
print(nombresDicc)
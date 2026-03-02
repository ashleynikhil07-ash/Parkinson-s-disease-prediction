import cv2
import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset_path = r"C:\NIKHIL\SEM - 4\hackathon\dataset"

X = []
y = []

for category in ["healthy", "parkinson"]:
    path = os.path.join(dataset_path, category)
    label = 0 if category == "healthy" else 1

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (300, 300))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            edge_count = np.sum(edges > 0)
            edge_density = edge_count / edges.size
            mean_intensity = np.mean(edges)
            std_dev = np.std(edges)
            edge_ratio = np.count_nonzero(edges) / edges.size

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = len(contours)

            if contour_count > 0:
                largest = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest)
            else:
                contour_area = 0

            X.append([
                edge_count, edge_density, mean_intensity,
                std_dev, edge_ratio, contour_count, contour_area
            ])
            y.append(label)

        except:
            pass

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, max_depth=10)
model.fit(X, y)

# SAVE MODEL
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model & scaler saved!")

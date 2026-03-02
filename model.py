import cv2
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

            X.append([edge_count, edge_density, mean_intensity, std_dev, edge_ratio])
            y.append(label)

        except:
            pass

X = np.array(X)
y = np.array(y)

print("Dataset Loaded:", len(X))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("🎯 Model Accuracy:", round(accuracy * 100, 2), "%")


test_image = r"C:\NIKHIL\SEM - 4\hackathon\dataset\healthy\V01HE02.png"

img = cv2.imread(test_image)
img = cv2.resize(img, (300, 300))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

edge_count = np.sum(edges > 0)
edge_density = edge_count / edges.size
mean_intensity = np.mean(edges)
std_dev = np.std(edges)
edge_ratio = np.count_nonzero(edges) / edges.size

features = np.array([[edge_count, edge_density, mean_intensity, std_dev, edge_ratio]])

prediction = model.predict(features)


if prediction == 1:
    print("🧠 Parkinson's Detected")
else:
    print("✅ Healthy")

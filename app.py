from flask import Flask, render_template, request, send_file, redirect, url_for, session
import cv2
import numpy as np
import pickle
import os
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import mysql.connector

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="welcome2nikhil",   
        database="parkinson_db"
    )

app = Flask(__name__)
app.secret_key = "super_secret_key"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# DOCTOR DATABASE
# -----------------------------
doctors = [
    {"name": "Dr. Kumar (Neurologist)", "phone": "9876543210", "lat": 10.7905, "lon": 78.7047},
    {"name": "Dr. Priya (Specialist)", "phone": "9123456780", "lat": 10.8000, "lon": 78.6900}
]

def get_nearest_doctor(user_lat, user_lon):
    min_dist = float("inf")
    nearest = None

    for doc in doctors:
        dist = (user_lat - doc["lat"])**2 + (user_lon - doc["lon"])**2
        if dist < min_dist:
            min_dist = dist
            nearest = doc

    return nearest


# -----------------------------
# LOGIN (PATIENT)
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form["name"]
        phone = request.form["phone"]

        conn = get_db()
        cursor = conn.cursor()

        # Check if user already exists
        cursor.execute("SELECT * FROM patients WHERE phone = %s", (phone,))
        user = cursor.fetchone()

        if user:
            # Existing user
            session["user"] = user[1]
            session["phone"] = user[2]
        else:
            # New user → insert into DB
            cursor.execute(
                "INSERT INTO patients (name, phone) VALUES (%s, %s)",
                (name, phone)
            )
            conn.commit()

            session["user"] = name
            session["phone"] = phone

        cursor.close()
        conn.close()

        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -----------------------------
# MAIN PAGE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    if "user" not in session:
        return redirect(url_for("login"))

    patient_name = session.get("user", "Unknown")
    patient_phone = session.get("phone", "N/A")

    result = score = risk = doctor_advice = image_path = None
    doctor_info = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            return "⚠️ No file selected"

        # Save image
        filename = str(datetime.now().timestamp()).replace(".", "")
        image_path = os.path.join(UPLOAD_FOLDER, f"{filename}.png")
        file.save(image_path)

        img = cv2.imread(image_path)
        if img is None:
            return "⚠️ Invalid image file"

        img = cv2.resize(img, (300, 300))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        gray_path = os.path.join(UPLOAD_FOLDER, f"{filename}_gray.png")
        edge_path = os.path.join(UPLOAD_FOLDER, f"{filename}_edges.png")

        cv2.imwrite(gray_path, gray)
        cv2.imwrite(edge_path, edges)

        # -----------------------------
        # FEATURE EXTRACTION
        # -----------------------------
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

        features = np.array([[edge_count, edge_density, mean_intensity,
                              std_dev, edge_ratio, contour_count, contour_area]])

        features = scaler.transform(features)

        prediction = model.predict(features)
        prob = model.predict_proba(features)[0][1]

        score = round(prob * 100, 2)

        # -----------------------------
        # RISK ANALYSIS
        # -----------------------------
        if score < 30:
            risk = "LOW"
            doctor_advice = "No immediate concern. Maintain healthy lifestyle."
        elif score < 70:
            risk = "MEDIUM"
            doctor_advice = "Consult a neurologist for further screening."
        else:
            risk = "HIGH"
            doctor_advice = "Immediate medical consultation recommended."

        result = "Parkinson's Detected" if prediction == 1 else "Healthy"

        # -----------------------------
        # LOCATION & DOCTOR
        # -----------------------------
        lat = request.form.get("lat")
        lon = request.form.get("lon")
        try:
            user_lat = float(lat) if lat else 0
            user_lon = float(lon) if lon else 0
        except:
                user_lat = 0
                user_lon = 0

        doctor_info = get_nearest_doctor(user_lat, user_lon)

        # -----------------------------
        # SAVE SESSION REPORT
        # -----------------------------
        session["report"] = {
            "patient": patient_name,
            "phone": patient_phone,
            "result": result,
            "score": score,
            "risk": risk,
            "doctor_advice": doctor_advice,
            "original": image_path,
            "gray": gray_path,
            "edges": edge_path,
            "edge_count": int(edge_count),
            "edge_density": round(edge_density, 4),
            "std_dev": round(std_dev, 2),
            "contour_count": int(contour_count),
            "doc_name": doctor_info["name"] if doctor_info else "N/A",
            "doc_phone": doctor_info["phone"] if doctor_info else "N/A"
        }

    return render_template("index.html",
                           result=result,
                           score=score,
                           risk=risk,
                           doctor=doctor_advice,
                           image=image_path,
                           doctor_info=doctor_info)


# -----------------------------
# DOWNLOAD PDF
# -----------------------------
@app.route("/download")
def download():
    data = session.get("report", {})

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    # Title
    content.append(Paragraph("Parkinson Detection Report", styles["Title"]))
    content.append(Spacer(1, 20))

    # Patient Info
    content.append(Paragraph(f"Patient Name: {data.get('patient')}", styles["Normal"]))
    content.append(Paragraph(f"Phone Number: {data.get('phone')}", styles["Normal"]))
    content.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))
    content.append(Spacer(1, 10))

    # Results
    content.append(Paragraph(f"Result: {data.get('result')}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {data.get('score')}%", styles["Normal"]))
    content.append(Paragraph(f"Risk Level: {data.get('risk')}", styles["Normal"]))
    content.append(Paragraph(f"Doctor Advice: {data.get('doctor_advice')}", styles["Normal"]))

    content.append(Spacer(1, 20))

    # Images
    if data.get("original"):
        content.append(Paragraph("Original Image", styles["Heading2"]))
        content.append(Image(data.get("original"), width=200, height=200))

    if data.get("gray"):
        content.append(Paragraph("Grayscale Image", styles["Heading2"]))
        content.append(Image(data.get("gray"), width=200, height=200))

    if data.get("edges"):
        content.append(Paragraph("Edge Detection", styles["Heading2"]))
        content.append(Image(data.get("edges"), width=200, height=200))

    content.append(Spacer(1, 20))

    # Features
    content.append(Paragraph("Extracted Features", styles["Heading2"]))
    content.append(Paragraph(f"Edge Count: {data.get('edge_count')}", styles["Normal"]))
    content.append(Paragraph(f"Edge Density: {data.get('edge_density')}", styles["Normal"]))
    content.append(Paragraph(f"Std Dev: {data.get('std_dev')}", styles["Normal"]))
    content.append(Paragraph(f"Contour Count: {data.get('contour_count')}", styles["Normal"]))

    content.append(Spacer(1, 20))

    # Doctor Info
    content.append(Paragraph("Recommended Doctor", styles["Heading2"]))
    content.append(Paragraph(f"Name: {data.get('doc_name')}", styles["Normal"]))
    content.append(Paragraph(f"Phone: {data.get('doc_phone')}", styles["Normal"]))

    doc.build(content)

    return send_file("report.pdf", as_attachment=True)

@app.route("/reset")
def reset():
    session.clear()
    return "Session Cleared"


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

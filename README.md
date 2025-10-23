# 🚗 Driver Drowsiness Detection System

### 🧠 *An AI-powered real-time system that monitors drivers’ eyes and mouth using webcam or phone camera to detect drowsiness and alert them instantly.*

---

## 🚀 Overview

The **Driver Drowsiness Detection System** is a Python-based **Flask Web Application** that detects a person’s drowsiness in real time using **OpenCV** and **Dlib**.  
It continuously tracks the driver’s **eye closure** and **yawning** patterns using either a **webcam** or a **mobile camera feed**.  
When it detects signs of fatigue, it triggers an **audio alarm** to instantly alert the user, preventing potential accidents.  

The project focuses on **road safety** through a **low-cost, real-time solution** that doesn’t require any special hardware — just a camera and Python.

---

## ⚙️ How the Project Works

This project is a **Drowsiness Detection System** built as a Python **Flask** web app. It combines a user-friendly web interface with computer vision processing in the backend.

### Step-by-step Workflow:

1. **Start the Server:**  
   Run `app.py` to launch the local Flask server.

2. **Open the Web App:**  
   Go to `http://127.0.0.1:5000` in a browser.

3. **Select Camera Source:**  
   Choose between:
   - **Run Using Webcam** – uses your laptop camera.  
   - **Run Using Phone Cam** – streams from your Android phone via IP.

4. **Real-time Detection:**  
   - Uses a pre-trained Dlib model `shape_predictor_68_face_landmarks.dat` to find **eyes** and **mouth**.
   - Calculates **Eye Aspect Ratio (EAR)** for eye closure.  
   - Calculates **Mouth Aspect Ratio (MAR)** for yawning detection.

5. **Alert System:**  
   - If **EAR** drops below threshold → Eyes closed → Audio Alert plays.  
   - If **MAR** rises above threshold → Yawn detected → Warning plays.

6. **Exit:**  
   Press `q` to quit the camera window and return to the web app.

---

## 🌐 Website Page Descriptions

### 🏠 1. Home Page  
Landing page with:
- A welcoming banner for the project.
- Three info cards:
  - **Causes of Drowsiness**
  - **Symptoms**
  - **Preventions**

### 🎯 2. Detect Drowsiness  
Main control page that lets the user:
- Start the detection.
- Choose between **webcam** or **phone camera** input.

### 📘 3. Understanding Drowsiness  
Educational page describing:
- **What causes drowsiness**
- **How it affects drivers**
- **Tips to prevent fatigue**

### 💤 4. Levels of Drowsiness  
Defines the detection levels:
- **Alert**
- **Slightly Drowsy**
- **Moderately Drowsy**
- **Very Drowsy**
- **Asleep**  

This page also mentions how the **EAR** value helps decide the drowsiness level.

### ⚙️ 5. How It Works  
Explains the full tech stack:
- **Backend:** Flask  
- **Frontend:** HTML, CSS, JavaScript  
- **Processing:** OpenCV, Dlib, Numpy, Imutils  
- Includes the overall workflow and architecture diagram.

---

## 💡 Motivation

Drowsy driving is a leading cause of traffic accidents worldwide.  
Traditional monitoring systems are expensive and limited to high-end vehicles.  
This project aims to **bring an affordable and accessible solution** using just a camera and open-source tools.

---

## 🏗 System Architecture

```text
          +-----------------------+
          |    Camera Input       |
          | (Webcam / Phone Cam)  |
          +----------+------------+
                     |
                     v
          +-----------------------+
          |  Face & Landmark Detection |
          | (Using dlib predictor .dat)|
          +----------+------------+
                     |
                     v
          +-----------------------+
          | EAR & MAR Calculation |
          | (Eye & Mouth Ratios)  |
          +----------+------------+
                     |
                     v
          +-----------------------+
          | Drowsiness Detection  |
          | (Threshold Comparison)|
          +----------+------------+
                     |
                     v
          +-----------------------+
          |  Audio Alert Trigger  |
          | (Using pygame mixer)  |
          +-----------------------+
````

---

## 🔬 Working Principle

1. **Facial Landmark Detection**
   Uses the `shape_predictor_68_face_landmarks.dat` model to locate facial key points.

2. **EAR (Eye Aspect Ratio)**
   Measures the eye openness; a drop below threshold signals closed eyes.

3. **MAR (Mouth Aspect Ratio)**
   Measures how wide the mouth opens; higher value indicates yawning.

4. **Audio Alert**
   Triggers using `pygame.mixer` to play warning sounds.

5. **Real-Time Display**
   Runs via Flask routes like `/video_feed`, streaming live video with detection overlays.

---

## 🧰 Technologies Used

| Library            | Purpose                                    |
| ------------------ | ------------------------------------------ |
| **OpenCV**         | Real-time image capture and processing     |
| **Dlib**           | Facial landmark detection                  |
| **Imutils**        | Simplifies frame resizing and manipulation |
| **Numpy**          | Numerical calculations for EAR and MAR     |
| **Pygame**         | Plays alert sounds                         |
| **Flask**          | Web framework for the application          |
| **Flask-SocketIO** | Enables live camera feed streaming         |
| **Requests**       | Fetches frames from Android IP camera      |

---

## 💻 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/theikram/Drowsiness-Detection-System.git
cd Drowsiness-Detection-System
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # For Mac/Linux
venv\Scripts\activate        # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download Dlib Model

Download [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
Extract it inside the project folder.

### 5️⃣ Run the App

```bash
python app.py
```

Then open your browser and visit:
👉 `http://127.0.0.1:5000`

---

## 🎯 Results

✅ Real-time detection of **eye closure** and **yawning**
✅ Instant **audio alerts** for safety
✅ Works with **webcam** or **Android IP camera**
✅ Simple web interface for all users

---

## ⚠️ Limitations & Future Scope

| Limitation                            | Future Improvement                                     |
| ------------------------------------- | ------------------------------------------------------ |
| Struggles in **low light** conditions | Add **infrared camera** or low-light enhancement       |
| May trigger alerts during talking     | Use **ML-based classifier** to improve accuracy        |
| Requires face facing the camera       | Add **multi-angle face detection**                     |
| No history or dashboard               | Add **data logging** or **driver analytics dashboard** |
| Works locally only                    | Deploy as **mobile app** or **IoT cloud service**      |

---

## 👨‍💻 Contributors

* **Ikram** — Developer & Researcher

---

## 🪪 License

This project is released under the [MIT License](LICENSE).

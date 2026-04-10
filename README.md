<h1 align="center">👤 Face Recognition Attendance System</h1>
<h3 align="center">Automated Attendance System using Face Recognition & Computer Vision</h3>

---

## 📌 Project Overview

A real-time attendance system that automatically detects and recognizes faces using webcam and marks attendance in a **CSV file** — no manual entry needed!

---

## ✨ Features

- 📸 Real-time face detection using webcam
- 🧠 Face recognition using HOG model
- ✅ Automatic attendance marking in CSV
- ⚡ Fast processing — downscaled frames for speed
- 🔒 Marks attendance only once per session
- ⚠️ Skips unrecognized faces as "UNKNOWN"
- 🖼️ Supports JPG, JPEG, PNG dataset images

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![face_recognition](https://img.shields.io/badge/face__recognition-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 📁 Project Structure

Face-Recognition-Attendance/
├── main.py                  # Main application
├── dataset/                 # Folder with face images (name.jpg)
├── attendance.csv           # Auto-generated attendance file
├── requirements.txt
└── README.md

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Akshat569-cmd/Face-Recognition-Attendance.git

# Install dependencies
pip install -r requirements.txt

# Add face images in dataset folder
# Image filename = Person's name (e.g., Akshat.jpg)

# Run the application
python main.py
```

---

## 📊 How It Works

Dataset folder se faces load aur encode hote hain
Webcam se real-time frame capture hota hai
Frame downscale hota hai faster processing ke liye
Face recognition model match dhundta hai
Match milne par attendance CSV mein mark hoti hai
'q' press karne par program band hota hai

---

## 👨‍💻 Developer

**Akshat Solanki** — AI/ML Developer
- 📧 akshatsolanki569@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/akshat-solanki-909578386)
- 🐙 [GitHub](https://github.com/Akshat569-cmd)

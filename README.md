# 🧵 Automated Real-Time Fabric Inspection using YOLOv8 and FastAPI

This project automates the conventional fabric inspection process — traditionally slow, manual, and error-prone — by combining **deep learning**, **computer vision**, and **web technologies** to deliver a **real-time, web-based fabric defect detection system**.

---

## 🚀 Overview

The system detects the following fabric defects:

* Holes
* Slubs
* Foreign yarn
* Surface contamination

We created a custom dataset called **RuRa**, which combines real-world defect data (collected in collaboration with MAS Holdings) and the public **TILDA** dataset. To address class imbalance, we applied various data augmentation techniques. Models including **VGG19**, **ResNet50**, and **YOLOv8** were trained, with **YOLOv8** achieving the highest performance.

---

## 🧠 Tech Stack

* **Model Training:** VGG19, ResNet50, YOLOv8
* **Dataset:** RuRa (MAS + TILDA), augmented to address class imbalance
* **Backend:** FastAPI + WebSocket for near real-time video stream processing
* **Frontend:** Responsive web UI with pause/resume functionality and defect reporting
* **Output:** Fault points, fault rate computation, and auto-generated inspection reports

---

## 🏆 Key Highlights

* ✅ Created and augmented a hybrid dataset (RuRa) for real-world defect scenarios
* ⚡ Achieved superior real-time detection with YOLOv8
* 🖥️ Built a live video streaming web interface with interactive controls
* 📊 Generated inspection reports with fault metrics and pass/fail status
* 📄 Published a research article documenting the methodology and results


🧵 Automated Real-Time Fabric Inspection using YOLOv8 and FastAPI

This project automates the conventional fabric inspection process — traditionally slow, manual, and error-prone — by combining deep learning, computer vision, and web technologies to deliver a real-time, web-based fabric defect detection system.

🚀 Overview

We developed a system capable of detecting fabric defects such as:

Holes
Slubs
Foreign yarn
Surface contamination
Using a custom dataset called RuRa, built from real factory data (collected with MAS Holdings) and the public TILDA dataset, we trained and evaluated multiple models including VGG19, ResNet50, and YOLOv8 — with YOLOv8 achieving the best performance.

🧠 Tech Stack

Model Training: VGG19, ResNet50, YOLOv8
Dataset: RuRa (MAS + TILDA), with class balancing via augmentation
Backend: FastAPI with WebSocket for live video streaming
Frontend: Responsive UI with live preview, pause/resume, and defect reporting
Output: Fault points, fault rate calculation, and auto-generated inspection report

🏆 Key Highlights
✅ Built and augmented a hybrid dataset (RuRa) for real-world defect scenarios
⚡ Achieved high-speed, accurate detection with YOLOv8 in live video streams
📊 Generated detailed quality reports to accept or reject fabric rolls
📄 Published a research article documenting the pipeline and results

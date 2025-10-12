# 💎 AI Diamond Classification System

This project implements an automated diamond quality grading system using Huawei MindSpore. The architecture follows a local client-server model: the server runs a FastAPI backend hosting the inference models, while the client is a desktop application deployed on a Raspberry Pi. The system processes diamond images to classify four quality parameters: Polish, Symmetry, Fluorescence, and Clarity using a multi-task learning approach.

---

## 📦 Overview

The system is designed to capture diamond images from the Raspberry Pi client, transmit them over a local network, and return quality grading results via a FastAPI server using a trained MindSpore model with EfficientNet backbone. The client application provides a simple interface for image upload and result display, while the server utilizes MindSpore Lite for efficient model inference.

---

## 📐 Architecture

- **Presentation Layer** (Raspberry Pi):  
  - Python desktop application with GUI
  - Allows image upload or camera capture for local inference  
  - Displays predicted quality grades and confidence scores  

- **Business Logic Layer** (Model Development & Server):  
  - Multi-task CNN model (EfficientNet-B0 backbone + 4 classification heads)  
  - Handles preprocessing, inference, and postprocessing logic  

- **Communication Layer**:  
  - REST API (HTTP POST) for cloud inference via **Huawei ModelArts**  
  - JSON response returns predicted grades and confidence values  

---

## 📄 License

This project is intended for educational and research use only.  
Model licenses are subject to [MindOCR's repository terms](https://github.com/mindspore-lab/mindocr).

---

## 👥 Contributors

Group01 — CPE180P E01 1T2526  
- [Beltran, Neil Ace Jefferson]  
- [Cacayan, Jacob Paolo]
- [Cali-at, Nevin John]  


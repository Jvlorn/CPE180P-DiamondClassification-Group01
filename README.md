# 💎 AI Diamond Classification System

This project implements a local client-server diamond quality grading system using **MindSpore**, **FastAPI**, and a **Flet-based client UI**. The system evaluates diamond images and predicts four quality attributes—**Polish**, **Symmetry**, **Fluorescence**, and **Clarity**—using a multi-task learning model.

---

## 📦 System Overview

**📂 Dataset Source**
The diamond image dataset used for training is publicly available on Kaggle and **is not included in this repository** to reduce storage size. You can download it from:

🔗 [https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset/data](https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset/data)

After downloading, place the dataset in the appropriate directory (e.g., `data/diamonds/`) before running `training.py`.

The system uses a local network setup with two main components, optimized for Raspberry Pi deployment:

* **Server (Backend):**

  * Built using FastAPI.
  * Loads a trained MindSpore model (`best_diamond_model.ckpt`).
  * Detects model architecture dynamically and loads the corresponding JSON label encoders.
  * Provides REST endpoints for health checks and image-based predictions.

* **Client (Frontend):**

  * Desktop application built with Flet (`client.py`).
  * Runs on a laptop or Raspberry Pi.
  * Allows users to upload diamond images, send them to the server, and visualize prediction results.

---

## 📁 Project Structure

```
.
├── client.py                  # Flet-based GUI client
├── server.py / server_fixed.py  # FastAPI backend with MindSpore integration
├── training.py                # Script to train the diamond classification model
├── generateEncoder.py         # Generates encoder JSON files for class labels
├── requirements.txt           # Python dependencies
├── diamond_data.csv           # Dataset label file (used during training)
├── best_diamond_model.ckpt    # Trained model checkpoint
├── polish_encoder.json        # JSON encoder for polish classes
├── symmetry_encoder.json      # JSON encoder for symmetry classes
├── fluorescence_encoder.json  # JSON encoder for fluorescence classes
├── clarity_encoder.json       # JSON encoder for clarity classes
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### ✅ 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ 2. Generate Label Encoders (If Missing)

```bash
python generateEncoder.py
```

### ✅ 3. Train the Model (Optional)

```bash
python training.py
```

This will generate a new `best_diamond_model.ckpt` file.

### ✅ 4. Start the Server

```bash
python server.py
```

The server will automatically:

* Detect the model architecture from the checkpoint.
* Load label encoders or create compatible ones.
* Serve prediction and health check endpoints.

### ✅ 5. Run the Client

```bash
python client.py
```

Enter the server address (e.g., `http://<Raspberry-Pi-IP>:8000`) and classify diamond images.

---

## 🌐 API Endpoints

| Method | Endpoint      | Description                              |
| ------ | ------------- | ---------------------------------------- |
| GET    | `/`           | Root endpoint; server status             |
| GET    | `/health`     | Checks if the model is loaded            |
| GET    | `/model-info` | Displays model & encoder details         |
| POST   | `/predict`    | Accepts an image and returns predictions |

---

## 📊 Prediction Output Format

The `/predict` endpoint returns a JSON response:

```json
{
  "success": true,
  "filename": "diamond.jpg",
  "predictions": {
    "polish": {
      "label": "Excellent",
      "confidence": 0.95,
      "all_probabilities": {
        "Excellent": 0.95,
        "Very Good": 0.03,
        "Good": 0.02
      }
    },
    "symmetry": { ... },
    "fluorescence": { ... },
    "clarity": { ... }
  }
}
```

---

## 🖥 Client Features

* Connect to FastAPI server.
* Upload or capture diamond images.
* Display prediction results with confidence levels.
* Color-coded confidence indicators (green, orange, red).

---

## 🚀 Future Improvements

* Add camera capture directly from Raspberry Pi.
* Support batch image processing.
* Enhance UI/UX with result history.
* Optimize training using data augmentation and EfficientNet.

---

## 📄 License

This project is for educational and research purposes only. Models and components are subject to the respective licenses of MindSpore and related repositories.

---

## 👥 Contributors

Group01 — CPE180P E01 1T2526

* Neil Ace Jefferson Beltran
* Jacob Paolo Cacayan
* Nevin John Cali-at

---

If you encounter issues or have suggestions, feel free to contribute or open an issue!

## ☁ Huawei ModelArts Deployment

### 📁 Preparing Files

* Upload `best_diamond_model.ckpt`, encoder JSON files, and `server.py` to an OBS (Object Storage Service) bucket.
* Ensure the same directory structure is preserved.

### 🚀 Deploying as an Inference Service

1. Log in to Huawei Cloud > **ModelArts**.
2. Navigate to **Model Management > Models > Create Model**.
3. Choose **Custom Image + MindSpore Runtime**.
4. Select the model files from OBS.
5. Deploy as a **Real-Time Inference Service** using FastAPI or MindSpore Serving.
6. After deployment, note the HTTPS endpoint (e.g., `https://xxxxxx.modelarts.com/v1/diamond/predict`).

### 🌐 Switching Client to Cloud

In the Flet client (client.py), update the server URL field to:

```
https://<modelarts-endpoint>/predict
```

No code change is required — just the URL input.

---

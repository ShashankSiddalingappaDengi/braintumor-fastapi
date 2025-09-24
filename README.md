🧠 Brain Tumor Detection with FastAPI + Vision Transformer + Docker
📌 Overview

This project provides a FastAPI application for brain tumor classification using a Vision Transformer (ViT) model.
It detects 4 categories from MRI scans:

Glioma

Meningioma

Pituitary

No Tumor

The API is built with FastAPI, containerized using Docker, and integrated with GitHub Actions CI/CD for automated builds and pushes to Docker Hub.

⚙️ Tech Stack

Python 3.10

FastAPI + Uvicorn (high-performance async API)

PyTorch + ViT (vit-pytorch)

Docker

GitHub Actions (CI/CD pipeline)

## 📂 Project Structure
braintumor-fastapi/
│── app.py # FastAPI app (prediction endpoint)
│── transformer.py # Vision Transformer model definition
│── requirements.txt # Dependencies
│── Dockerfile # Docker build instructions
│── brain_tumer.ipynb # Training notebook (optional)
│── .github/workflows/ # GitHub Actions pipeline
│── README.md

🚀 Run FastAPI Locally
1️⃣ Clone the repo
git clone https://github.com/ShashankSiddalingappaDengi/braintumor-fastapi.git
cd braintumor-fastapi

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Download model weights

Download the trained model from [Google Drive / Hugging Face link]
Place it inside models/:

braintumor-fastapi/
└── models/
    └── best_model.pth

4️⃣ Run FastAPI app
uvicorn app:app --reload


Now open 👉 http://127.0.0.1:8000/docs

FastAPI automatically generates interactive Swagger UI where you can upload MRI images and see predictions.

🐳 Run with Docker
1️⃣ Build the image
docker build -t braintumor-fastapi .

2️⃣ Run the container
docker run -p 8000:8000 braintumor-fastapi


FastAPI API available at 👉 http://localhost:8000/docs

🐳 Pull Pre-Built Image from Docker Hub
docker pull shashankdengi/braintumor-fastapi:latest
docker run -p 8000:8000 shashankdengi/braintumor-fastapi:latest

📸 Example Usage
Swagger UI

Visit: http://127.0.0.1:8000/docs

Upload MRI image → get prediction instantly.

Python Test Script
import requests

url = "http://127.0.0.1:8000/predict"
file = {"file": open("sample_mri.jpg", "rb")}
response = requests.post(url, files=file)
print(response.json())

🔄 CI/CD with GitHub Actions

On every push to main:

Dependencies installed

Docker image built

Docker image pushed to Docker Hub → shashankdengi/braintumor-fastapi
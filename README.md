# 🌿 Plant Disease Prediction System

A web-based application that uses a Convolutional Neural Network (CNN) to identify plant diseases from leaf images. Built with Python and Streamlit, this tool is designed to assist farmers and agriculturists in early disease detection and prevention.

---


## 📷 Demo

[![Watch the video](https://img.youtube.com/vi/CFBJPW3bHKs/0.jpg)](https://www.youtube.com/watch?v=CFBJPW3bHKs)

---

## 💡 Features

- ✅ Image-based disease detection
- ✅ Convolutional Neural Network (CNN) for classification
- ✅ Streamlit-based web interface for ease of use
- ✅ Real-time predictions

---


## 🧠 Model Details

- **Model Type:** CNN (Convolutional Neural Network)
- **Framework:** TensorFlow / Keras
- **Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Classes:** Includes multiple classes of healthy and diseased plant leaves

---

## 🛠️ Tech Stack

| Component      | Technology       |
|----------------|------------------|
| Machine Learning | TensorFlow 2.18.0 / Keras |
| Web App        | Streamlit         |
| Language       | Python 3.10.5        |
| Deployment     | Localhost |

---


## ⚙️ Installation

## ⚠️ Important Compatibility Note

> **Please ensure the following versions are used to avoid compatibility issues:**
>
> - **Python:** `3.10.5`  
> - **TensorFlow:** `2.18.0`
>
> Using other versions of Python or TensorFlow may result in errors during installation or runtime.

---

To run this project locally:

1. Clone the repository:

```bash
git clone https://github.com/yourusername/plant-disease-prediction.git
cd plant-disease-prediction
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run main.py
```

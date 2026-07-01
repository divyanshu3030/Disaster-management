# 🌍 Disaster Prediction System using LSTM

An AI-powered **Disaster Prediction System** built using **Python, TensorFlow, LSTM, HTML, CSS, and JavaScript**. The project analyzes historical disaster records and predicts the expected number of disasters for an upcoming month using a Long Short-Term Memory (LSTM) neural network.

The application also includes a responsive frontend with a modern landing page that allows users to access the prediction module.

---

# 📌 Project Overview

This project leverages **Deep Learning (LSTM)** to forecast future disaster occurrences based on historical monthly disaster data. The system preprocesses the dataset, trains an LSTM model, and predicts the number of disasters for a selected state or for the entire country.

---

# ✨ Features

* 🌍 State-wise Disaster Prediction
* 🧠 LSTM Deep Learning Model
* 📊 Monthly Disaster Forecasting
* 📈 Time Series Prediction
* 📂 CSV Dataset Processing
* ⚡ Automatic Data Cleaning
* 📉 Min-Max Feature Scaling
* 📱 Responsive Landing Page
* 🎨 Modern UI Design
* 🔮 Predict Next Month's Disaster Count

---

# 🛠️ Technologies Used

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend / AI

* Python
* TensorFlow
* Keras
* Pandas
* NumPy
* Scikit-learn

---

# 📚 Machine Learning Workflow

1. Load Disaster Dataset
2. Clean & Preprocess Data
3. Group Monthly Disaster Records
4. Scale Data using MinMaxScaler
5. Create Time-Series Sequences
6. Build LSTM Neural Network
7. Train Model
8. Predict Future Disaster Count
9. Display Prediction Results

---

# 📂 Project Structure

```text
Disaster-Prediction-System/
│
├── index.html
├── disaster.html
├── style.css
├── script.js
│
├── disaster_prediction.py
├── Disaster_Management_AI_Dataset.csv
│
├── README.md
└── requirements.txt
```

---

# 📊 Dataset Information

The dataset contains historical disaster records including:

* Date
* State
* Disaster Type
* Monthly Disaster Count

The data is grouped by month to create a time-series dataset for LSTM training.

---

# 🧠 Deep Learning Model

## Long Short-Term Memory (LSTM)

The project uses an **LSTM Neural Network**, which is highly effective for time-series forecasting.

### Model Architecture

* LSTM Layer (50 Units)
* Dropout Layer (0.2)
* LSTM Layer (50 Units)
* Dropout Layer (0.2)
* Dense Output Layer

### Optimizer

* Adam Optimizer

### Loss Function

* Mean Squared Error (MSE)

---

# 🌍 User Workflow

1. Open the landing page.
2. Click **Simulate Prediction**.
3. Enter:

   * State Name
   * Target Month (optional)
4. The LSTM model processes historical data.
5. The system predicts the expected disaster count.
6. Results display:

   * Region
   * Last Available Data
   * Predicted Month
   * Predicted Disaster Count
   * Most Common Recent Disaster Types

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/disaster-prediction-system.git
```

Go to the project folder

```bash
cd disaster-prediction-system
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python disaster_prediction.py
```

---

# 📦 Required Libraries

* TensorFlow
* Keras
* Pandas
* NumPy
* Scikit-learn

Install manually if needed:

```bash
pip install tensorflow pandas numpy scikit-learn
```

---

# 📈 Prediction Output

The system provides:

* 🌍 Selected Region
* 📅 Last Available Data Month
* 🔮 Predicted Month
* 📊 Predicted Disaster Count
* ⚠️ Recent Disaster Types
* ✅ Prediction Status

---

# 🎯 Current Features

* ✅ Responsive Landing Page
* ✅ Time-Series Forecasting
* ✅ LSTM Deep Learning Model
* ✅ Monthly Disaster Prediction
* ✅ Automatic Data Preprocessing
* ✅ State-wise Forecasting
* ✅ Dataset Scaling
* ✅ Interactive Console Input

---

# 🔮 Future Improvements

* 🌐 Flask/Django Web Integration
* ☁️ Cloud Deployment
* 📱 Mobile Application
* 📊 Dashboard & Data Visualization
* 🔔 Real-Time Disaster Alert System
* 🌎 Multi-country Prediction Support

---

<img width="1218" height="768" alt="Screenshot 2026-07-01 111045" src="https://github.com/user-attachments/assets/136e33a0-4d7c-4af3-b934-9beaeb2f1954" />

<img width="1347" height="887" alt="Screenshot 2026-07-01 111052" src="https://github.com/user-attachments/assets/cad899d0-120f-49a1-acb3-613e3d1a8885" />


# 👨‍💻 Developed By

**Divyanshu Negi**

---

# ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

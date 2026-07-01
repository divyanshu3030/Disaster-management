🌍 Disaster Prediction System using LSTM

An AI-powered Disaster Prediction System built using Python, TensorFlow, LSTM, HTML, CSS, and JavaScript. The project analyzes historical disaster records and predicts the expected number of disasters for an upcoming month using a Long Short-Term Memory (LSTM) neural network.

The application also includes a responsive frontend with a modern landing page that allows users to access the prediction module.

📌 Project Overview

This project leverages Deep Learning (LSTM) to forecast future disaster occurrences based on historical monthly disaster data. The system preprocesses the dataset, trains an LSTM model, and predicts the number of disasters for a selected state or for the entire country.

✨ Features
🌍 State-wise Disaster Prediction
🧠 LSTM Deep Learning Model
📊 Monthly Disaster Forecasting
📈 Time Series Prediction
📂 CSV Dataset Processing
⚡ Automatic Data Cleaning
📉 Min-Max Feature Scaling
📱 Responsive Landing Page
🎨 Modern UI Design
🔮 Predict Next Month's Disaster Count
🛠️ Technologies Used
Frontend
HTML5
CSS3
JavaScript
Backend / AI
Python
TensorFlow
Keras
Pandas
NumPy
Scikit-learn
📚 Machine Learning Workflow
Load Disaster Dataset
Clean & Preprocess Data
Group Monthly Disaster Records
Scale Data using MinMaxScaler
Create Time-Series Sequences
Build LSTM Neural Network
Train Model
Predict Future Disaster Count
Display Prediction Results
📂 Project Structure
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
📊 Dataset Information

The dataset contains historical disaster records including:

Date
State
Disaster Type
Monthly Disaster Count

The data is grouped by month to create a time-series dataset for LSTM training.

🧠 Deep Learning Model
Long Short-Term Memory (LSTM)

The project uses an LSTM Neural Network, which is highly effective for time-series forecasting.

Model Architecture
LSTM Layer (50 Units)
Dropout Layer (0.2)
LSTM Layer (50 Units)
Dropout Layer (0.2)
Dense Output Layer
Optimizer
Adam Optimizer
Loss Function
Mean Squared Error (MSE)
🌍 User Workflow
Open the landing page.
Click Simulate Prediction.
Enter:
State Name
Target Month (optional)
The LSTM model processes historical data.
The system predicts the expected disaster count.
Results display:
Region
Last Available Data
Predicted Month
Predicted Disaster Count
Most Common Recent Disaster Types
🚀 Installation

Clone the repository

git clone https://github.com/yourusername/disaster-prediction-system.git

Go to the project folder

cd disaster-prediction-system

Install dependencies

pip install -r requirements.txt

Run the project

python disaster_prediction.py
📦 Required Libraries
TensorFlow
Keras
Pandas
NumPy
Scikit-learn

Install manually if needed:

pip install tensorflow pandas numpy scikit-learn
📈 Prediction Output

The system provides:

🌍 Selected Region
📅 Last Available Data Month
🔮 Predicted Month
📊 Predicted Disaster Count
⚠️ Recent Disaster Types
✅ Prediction Status
🎯 Current Features
✅ Responsive Landing Page
✅ Time-Series Forecasting
✅ LSTM Deep Learning Model
✅ State-wise Forecasting
✅ Dataset Scaling
✅ Interactive Console Input

🔮 Future Improvements
🌐 Flask/Django Web Integration
📍 Live Weather API Integration

👨‍💻 Developed By

Divyanshu Negi

⭐ Support

If you found this project useful, consider giving it a ⭐ Star on GitHub.

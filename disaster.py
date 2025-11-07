import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------
DATASET_PATH = "Disaster_Management_AI_Dataset.csv"
LOOKBACK = 12 # Number of previous months to look at for prediction
EPOCHS = 50   # Training epochs (kept low for fast execution)
BATCH_SIZE = 1 # Training batch size

# -----------------------------------------------------------
# Model Architecture Definition
# -----------------------------------------------------------
def create_lstm_model(lookback):
    """Defines and compiles the LSTM model."""
    model = Sequential([
        # LSTM layer 1: Processes sequences, returns output for next layer
        LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        
        # LSTM layer 2: Final processing layer
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        # Dense layer: Output layer for the single predicted value
        Dense(units=1) 
    ])
    
    # Compile the model using Adam optimizer and Mean Squared Error loss
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------------------------------------------
# Data Preparation Functions
# -----------------------------------------------------------

def prepare_data_for_lstm(df, lookback=12):
    """
    Prepares time series data by scaling and creating sequences (X and y).
    Returns: scaled_data, X (features), y (target), scaler instance
    """
    # 1. Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    # We only scale the 'count' column
    scaled = scaler.fit_transform(df['count'].values.reshape(-1, 1))

    # 2. Creating sequences (X and y)
    X, y = [], []
    # Loop from 'lookback' month to the end of the data
    for i in range(lookback, len(scaled)):
        # X: lookback previous months (e.g., 12)
        X.append(scaled[i - lookback:i, 0])
        # y: the current month's value (the target)
        y.append(scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshaping X for Keras LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return scaled, X, y, scaler

def process_disaster_data(state_name="India", target_month_str=None, lookback=LOOKBACK):
    """
    Loads, cleans, aggregates, trains the LSTM model, and predicts the count 
    for the specified state and target month.
    """
    try:
        # Load and clean data
        data = pd.read_csv(DATASET_PATH)
        data.columns = data.columns.str.strip().str.replace(' ', '_').str.lower()
        data['date'] = pd.to_datetime(data['date'])
    except Exception as e:
        return {"error": f"Failed to load or process CSV: {e}"}

    # Filter data by state
    if state_name.lower() != "india":
        df = data[data['state'].str.lower() == state_name.lower()]
    else:
        df = data.copy()

    if df.empty:
        return {"error": f"No data found for region: {state_name}"}

    # Monthly Grouping: Count of all disasters per month
    monthly = df.groupby(pd.Grouper(key='date', freq='M')).size().reset_index(name='count')
    monthly = monthly.sort_values('date').reset_index(drop=True)
    
    if len(monthly) < lookback + 1:
        return {"error": f"Not enough data points ({len(monthly)}) for a {lookback}-month lookback required for LSTM."}

    # Prepare data for LSTM: Scaling and sequence creation
    scaled, X, y, scaler = prepare_data_for_lstm(monthly, lookback)

    # ------------------------------------------------------------
    # STEP 1: TRAIN THE LSTM MODEL
    # ------------------------------------------------------------
    model = create_lstm_model(lookback)
    
    # Train the model
    print(f"Training LSTM for {state_name.title()} with {X.shape[0]} samples...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    print("Training complete.")

    # ------------------------------------------------------------
    # STEP 2: PREPARE INPUT FOR NEXT MONTH PREDICTION
    # ------------------------------------------------------------
    # Use the last 'lookback' (12) points from the scaled data to predict the future.
    last_12_months = scaled[-lookback:].reshape(1, lookback, 1) # Reshape for prediction [1, 12, 1]

    # ------------------------------------------------------------
    # STEP 3: PREDICT THE NEXT VALUE
    # ------------------------------------------------------------
    scaled_prediction = model.predict(last_12_months, verbose=0)
    
    # Inverse transform to get the actual count
    predicted_count_float = scaler.inverse_transform(scaled_prediction)[0, 0]
    predicted_count = max(1, round(predicted_count_float)) # Ensure count is at least 1

    # ------------------------------------------------------------
    # Determine Prediction Date
    # ------------------------------------------------------------
    last_date = monthly['date'].max()
    
    if target_month_str:
        # User provided a specific month (e.g., '2025-10')
        parts = target_month_str.split('-')
        target_date = pd.to_datetime(f"{parts[0]}-{parts[1]}-01") + pd.DateOffset(days=14) # Center date
    else:
        # Predict the month immediately following the last data point
        target_date = last_date + pd.DateOffset(months=1)

    if target_date <= last_date:
        return {"error": f"Target month ({target_date.strftime('%B %Y')}) must be after the last available data date ({last_date.strftime('%B %Y')})."}

    # Find most common recent disaster type
    disaster_col = next((col for col in data.columns if 'disaster' in col and 'type' in col), None)
    
    recent_disasters = []
    if disaster_col in df.columns:
          # Get top 3 most common disaster types in the filtered data
         recent_disasters = df.sort_values('date', ascending=False)[disaster_col].value_counts().head(3).index.tolist()
         
    # The backend would typically return a JSON object
    return {
        "state": state_name.title(),
        "last_data_date": last_date.strftime('%B %Y'),
        "next_prediction_date": target_date.strftime('%B %Y'),
        "predicted_count": predicted_count,
        "recent_disaster_types": recent_disasters,
        "prediction_status": "LSTM Model Prediction Complete"
    }

# -----------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------
if __name__ == "__main__":
    # Ensure TensorFlow is working
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Ask the user for the state name and target month
    user_state = input("Enter State name (e.g., Maharashtra, or 'India'): ").strip()
    user_month = input("Enter Target Month (YYYY-MM, or leave blank for next month): ").strip()
    
    # Start data processing and prediction
    summary = process_disaster_data(user_state, user_month if user_month else None)

    print(f"\n--- AI LSTM Prediction for {user_state.title()} ---")
    
    if 'error' in summary:
        print(f"⚠️ Error: {summary['error']}")
    else:
        print(f"Region: {summary['state']}")
        print(f"Data Up to: {summary['last_data_date']}")
        print(f"Predicted Disaster Count for {summary['next_prediction_date']}: {summary['predicted_count']}")
        print(f"Top Recent Disasters: {', '.join(summary['recent_disaster_types'])}")
        print(f"\n✅ {summary['prediction_status']}")
        
        # Display model summary (optional, for debugging)
        # create_lstm_model(LOOKBACK).summary()

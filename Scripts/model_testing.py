import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'Model', 'mental_health_model.h5')
model = load_model(model_path)

scaler_path = os.path.join(os.path.dirname(__file__), '..', 'Model', 'scaler.pkl')
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

def predict_mental_health(new_data):
    new_data_scaled = scaler.transform(new_data)
    
    prediction = model.predict(new_data_scaled)
    
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class, prediction

def test_model(test_cases):
    for i, case in enumerate(test_cases, 1):
        class_prediction, probabilities = predict_mental_health(np.array([case]))
        print(f"Test Case {i}:")
        print(f"  Input: {case}")
        print(f"  Predicted class: {'Normal' if class_prediction[0] == 0 else 'Depression'} (Class: {class_prediction[0]})")
        print(f"  Class probabilities: {probabilities[0]}")
        print()

# Defined 5 different static inputs for testing
test_cases = [
    # Example 1: Young male, moderate stress, average performance
    [0, 24.0, 2.0, 5.9, 5.0, 1, 2, 0, 3.0, 2.0, 1, 0.0],
    # Example 2: Older female, high stress, poor performance
    [1, 35.0, 4.0, 3.5, 2.0, 0, 1, 1, 12.0, 4.0, 1, 1.0],
    # Example 3: Middle-aged male, low stress, good performance
    [0, 45.0, 1.0, 8.0, 4.0, 2, 0, 0, 8.0, 1.0, 0, 0.0],
    # Example 4: Young female, very high stress, excellent performance
    [1, 22.0, 5.0, 9.5, 3.0, 1, 1, 0, 15.0, 3.0, 1, 2.0],
    # Example 5: Older male, extreme stress, very poor performance
    [0, 50.0, 5.0, 2.0, 1.0, 0, 1, 1, 10.0, 5.0, 1, 1.0]
]

if __name__ == "__main__":
    test_model(test_cases)
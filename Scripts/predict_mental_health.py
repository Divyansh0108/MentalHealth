import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

model = load_model('../Model/mental_health_model.h5')

with open('../Model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def predict_mental_health(new_data):
    new_data_scaled = scaler.transform(new_data)
    
    prediction = model.predict(new_data_scaled)
    
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class, prediction


if __name__ == "__main__":
    
    new_symptoms = np.array([[1,24.0,2.0,5.9,5.0,1,2,0,3.0,2.0,1,0.0]]) 
    
    class_prediction, probabilities = predict_mental_health(new_symptoms)
    print(f"Predicted class: {class_prediction[0]}")
    print(f"Class probabilities: {probabilities[0]}")
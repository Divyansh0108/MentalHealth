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

def get_user_input():
    print("Please enter the following details for mental health prediction:")
    
    gender = int(input("Gender (0 for Male, 1 for Female): "))
    age = float(input("Age: "))
    academic_pressure = float(input("Academic Pressure (1-5): "))
    cgpa = float(input("CGPA (0.0 - 10.0): "))
    study_satisfaction = float(input("Study Satisfaction (1-5): "))
    sleep_duration = int(input("Sleep Duration (0 for less than 5h, 1 for 5-6h, 2 for 7-8h, 3 for more than 8h): "))
    dietary_habits = int(input("Dietary Habits (0 for Healthy, 1 for Unhealthy, 2 for Moderate): "))
    suicidal_thoughts = int(input("Suicidal Thoughts (0 for No, 1 for Yes): "))
    work_study_hours = float(input("Work/Study Hours per day: "))
    financial_stress = float(input("Financial Stress (1-5): "))
    family_history = int(input("Family History of Mental Illness (0 for No, 1 for Yes): "))
    education_level = int(input("Education Level (0 for Graduated, 1 for Post Graduated, 2 for Higher Secondary): "))
    
    return np.array([[gender, age, academic_pressure, cgpa, study_satisfaction, 
                    sleep_duration, dietary_habits, suicidal_thoughts, 
                    work_study_hours, financial_stress, family_history, 
                    education_level]])

if __name__ == "__main__":
    
    new_symptoms = get_user_input()
    
    class_prediction, probabilities = predict_mental_health(new_symptoms)
    print(f"Predicted class: {class_prediction[0]}")
    print(f"Class probabilities: {probabilities[0]}")
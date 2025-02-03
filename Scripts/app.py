import numpy as np
import streamlit as st
import google.generativeai as genai
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from predict_mental_health import predict_mental_health

genai.configure(api_key=st.secrets["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

mental_health_model = load_model('../Notebooks/mental_health_model.h5')

sleep_mapping = {
    'Less than 5 hours': {'value': 0, 'risk': 'High Sleep Disruption 😴', 'impact': 'Severe negative impact on mental health'},
    '5-6 hours': {'value': 1, 'risk': 'Moderate Sleep Disruption 😴', 'impact': 'Potential cognitive and emotional challenges'},
    '7-8 hours': {'value': 2, 'risk': 'Optimal Sleep 😊', 'impact': 'Healthy sleep pattern'},
    'More than 8 hours': {'value': 3, 'risk': 'Potential Oversleeping 😴', 'impact': 'May indicate underlying mental health concerns'}
}

diet_mapping = {
    'Healthy': {'value': 0, 'risk': 'Low Nutritional Risk 🍏', 'impact': 'Supports overall mental well-being'},
    'Unhealthy': {'value': 1, 'risk': 'High Nutritional Risk 🍔', 'impact': 'Potential negative impact on mental health'},
    'Moderate': {'value': 2, 'risk': 'Moderate Nutritional Concerns 🥗', 'impact': 'Some areas for dietary improvement'}
}

degree_value_mapping = {
    'Graduated': 0,
    'Post Graduated': 1,
    'Higher Secondary': 2
}

def generate_detailed_explanation(prediction, symptoms, feature_details, response_level='detailed'):
    feature_insights = []
    risk_score = 0

    if sleep_mapping[feature_details['sleep_duration']]['risk'] != 'Optimal Sleep 😊':
        feature_insights.append(f"Sleep pattern shows {sleep_mapping[feature_details['sleep_duration']]['risk']}. {sleep_mapping[feature_details['sleep_duration']]['impact']}.")
        risk_score += 1
    
    if symptoms[2] > 3:
        feature_insights.append("High academic pressure detected 📚. This can significantly impact mental well-being.")
        risk_score += 2
    
    if symptoms[3] < 5.0:
        feature_insights.append("Lower academic performance may be contributing to stress and anxiety 📉.")
        risk_score += 1
    
    if diet_mapping[feature_details['dietary_habits']]['risk'] != 'Low Nutritional Risk 🍏':
        feature_insights.append(f"Dietary habits indicate {diet_mapping[feature_details['dietary_habits']]['risk']}. {diet_mapping[feature_details['dietary_habits']]['impact']}.")
        risk_score += 1
    
    if feature_details['suicidal_thoughts'] == 'Yes':
        feature_insights.append("Presence of suicidal thoughts requires immediate professional attention 🚨.")
        risk_score += 3
    
    if symptoms[9] > 3:
        feature_insights.append("High financial stress detected 💸, which can significantly impact mental health.")
        risk_score += 2

    if prediction == 0:
        if response_level == 'concise':
            overall_assessment = "Mental health appears stable 🔄."
            risk_level = "Low"
            color = "green"
        else:
            overall_assessment = "While your current mental health appears stable 🔄, there are several areas that could benefit from attention and support."
            risk_level = "Low to Moderate"
            color = "green"
    else:
        if response_level == 'concise':
            overall_assessment = "Symptoms suggest potential depression ☹️."
            risk_level = "High"
            color = "red"
        else:
            overall_assessment = "Your symptoms suggest you might be experiencing depression ☹️. Professional support can help you navigate these challenges."
            risk_level = "High"
            color = "red"

    prompt = f"User Profile Details:\n" \
             f"Current Mental Health Prediction: {'Normal' if prediction == 0 else 'Depression'}\n" \
             f"Key Insights: {' '.join(feature_insights)}\n\n" \
             f"Provide a {'brief' if response_level == 'concise' else 'comprehensive'}, empathetic guidance focusing on mental health support and potential interventions."

    try:
        response = model.generate_content(prompt)
        ai_insights = response.text
    except Exception as e:
        ai_insights = "Professional consultation recommended 🤝." if response_level == 'concise' else "Unable to generate insights. Professional consultation recommended 🤝."

    return {
        'feature_insights': feature_insights,
        'overall_assessment': overall_assessment,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'color': color,
        'ai_insights': ai_insights
    }

def explain_with_shap(symptoms):
    symptoms = np.array(symptoms).reshape(1, -1)
    
    try:
        background = np.zeros_like(symptoms)
        explainer = shap.DeepExplainer(mental_health_model, background)
        shap_values = explainer.shap_values(symptoms)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        shap.summary_plot(shap_values[0], symptoms[0], plot_type="bar", show=False)
        plt.title("Feature Importance in Mental Health Prediction 📊")
        
        plt.subplot(2, 1, 2)
        shap.summary_plot(shap_values[0], symptoms[0], show=False)
        plt.title("Detailed Feature Impact Visualization 🔍")
        
        plt.tight_layout()
        plt.savefig('shap_plot_detailed.png')
        plt.close()
        
        return 'shap_plot_detailed.png'
    
    except Exception as e:
        st.error(f"Error in SHAP explanation: {e}")
        return None

def main():
    st.title('Comprehensive Mental Health Predictor and Advisor 🧠')
    st.markdown("### Holistic Mental Health Assessment 🌱")

    gender = st.radio("Gender", ['Male', 'Female'])
    age = st.slider("Age", min_value=18, max_value=100, value=24)
    academic_pressure = st.slider("Academic Pressure", min_value=1, max_value=5, value=2)
    cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=5.9)
    study_satisfaction = st.slider("Study Satisfaction", min_value=1, max_value=5, value=3)
    sleep_duration = st.selectbox("Sleep Duration", ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'])
    dietary_habits = st.selectbox("Dietary Habits", ['Healthy', 'Unhealthy', 'Moderate'])
    suicidal_thoughts = st.radio("Suicidal Thoughts", ['No', 'Yes'])
    work_study_hours = st.slider("Work/Study Hours", min_value=0, max_value=24, value=8)
    financial_stress = st.slider("Financial Stress", min_value=1, max_value=5, value=2)
    family_history = st.radio("Family History of Mental Illness", ['No', 'Yes'])
    new_degree = st.selectbox("New Degree", ['Graduated', 'Post Graduated', 'Higher Secondary'])

    col1, col2 = st.columns(2)
    
    with col1:
        quick_insights = st.button('Quick Insights ⚡')
    
    with col2:
        detailed_analysis = st.button('Detailed Analysis 🔍')

    response_level = 'concise' if quick_insights else 'detailed'

    if quick_insights or detailed_analysis:
        symptoms = [
            0 if gender == 'Male' else 1,
            age,
            academic_pressure,
            cgpa,
            study_satisfaction,
            sleep_mapping[sleep_duration]['value'],
            diet_mapping[dietary_habits]['value'],
            1 if suicidal_thoughts == 'Yes' else 0,
            work_study_hours,
            financial_stress,
            1 if family_history == 'Yes' else 0,
            degree_value_mapping[new_degree]
        ]
        
        feature_details = {
            'gender': gender,
            'age': age,
            'academic_pressure': academic_pressure,
            'cgpa': cgpa,
            'study_satisfaction': study_satisfaction,
            'sleep_duration': sleep_duration,
            'dietary_habits': dietary_habits,
            'suicidal_thoughts': suicidal_thoughts,
            'work_study_hours': work_study_hours,
            'financial_stress': financial_stress,
            'family_history': family_history,
            'new_degree': new_degree
        }
        
        symptoms_array = np.array(symptoms).reshape(1, -1)
        
        class_prediction, _ = predict_mental_health(symptoms_array)
        explanation = generate_detailed_explanation(class_prediction[0], symptoms, feature_details, response_level)
        
        st.markdown("### Mental Health Assessment Result 📊")
        st.markdown(f"**Risk Level:** <span style='color:{explanation['color']}'>{explanation['risk_level']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Risk Score:** {explanation['risk_score']} / 10")
        
        if response_level == 'detailed':
            st.markdown("#### Key Insights:")
            for insight in explanation['feature_insights']:
                st.markdown(f"- {insight}")
            
            st.markdown("#### Overall Assessment:")
            st.write(explanation['overall_assessment'])
            
            st.markdown("#### Personalized Guidance:")
            st.write(explanation['ai_insights'])
            
            st.markdown("#### Model Interpretation 📈")
            shap_plot = explain_with_shap(symptoms)
            if shap_plot:
                st.image(shap_plot, caption="Detailed Feature Impact Analysis")
            
            st.markdown("### Recommended Next Steps 📋")
            st.markdown("""
            1. **Professional Consultation:** Consider speaking with a mental health professional 🩺
            2. **Self-Care:** Focus on sleep, nutrition, and stress management 🌿
            3. **Support Network:** Reach out to trusted friends and family 🤝
            4. **Regular Monitoring:** Continue tracking your mental health indicators 🔄
            """)
        else:
            st.markdown("#### Quick Guidance:")
            st.write(explanation['ai_insights'])

if __name__ == "__main__":
    main()
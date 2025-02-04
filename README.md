# Mental Health Predictor 🧠

## Name of the Project
Mental Health Predictor and Advisor

## What the Project Does
This project provides a tool for predicting and assessing mental health status, particularly focusing on depression among students. It uses a machine learning model to predict the likelihood of depression based on various input features and offers personalized insights and recommendations for mental health management. 📊🧘‍♀️

## Motive of the Project
The primary motive is:
- To aid in the early detection of mental health issues, especially depression among students. 🔍
- To provide educational institutions, counselors, and individuals with tools for proactive mental health management. 🏫🤝
- To foster awareness and encourage a preventive approach to mental health care. 🌱

## Tools Used
- **Python**: For scripting and data manipulation. 🐍
- **TensorFlow**: For building and training the neural network model. 🧑‍💻
- **Streamlit**: For creating an interactive web application frontend. 🌐
- **Google Generative AI**: For generating personalized insights and recommendations. 💡
- **Matplotlib**: For data visualization. 📈
- **LIME**: For model interpretability and explaining predictions. 🔍
- **scikit-learn**: For machine learning utilities including data preprocessing. ⚙️

## Dataset Used
- **Filename**: Cleaned_dataset.csv
- **Source**: Custom student mental health dataset.

### Dataset Details
- **Initial Records**: 27,000 rows
- **Initial Columns**: 21 columns
- **Format**: CSV
- **Columns**: 
  - ID: Unique identifier for each student. 🆔
  - Age: Age of the student. 📅
  - Gender: Gender of the student (e.g., Male, Female). 🚻
  - City: Geographic location. 🌍
  - CGPA: Cumulative Grade Point Average. 📚
  - Sleep Duration: Average daily sleep duration. 😴
  - Profession: Student's current or intended profession. 👤
  - Work Pressure: Level of pressure from work or studies. 💼
  - Academic Pressure: Academic stress level. 📖
  - Study Satisfaction: Satisfaction with studies. 😊
  - Job Satisfaction: Satisfaction with job (if applicable). 👨‍💼👩‍💼
  - Dietary Habits: Description of dietary habits. 🍎
  - **Target Variable**: Depression_Status (Binary - Yes/No indicating depression status) ☹️/🙂

### About the Dataset
- **Purpose**: To analyze factors contributing to depression among students. 🔬
- **Ethical Considerations**: Data anonymization, privacy, and consent were strictly adhered to in collecting and using this dataset. 🔒

## Findings
- Key features like sleep duration, academic pressure, and dietary habits significantly influence mental health outcomes. 🌟
- The model shows high accuracy in predicting depression likelihood based on these features. 🎯

## Models Tested
- Logistic Regression
- Random Forest 🌳
- Neural Networks 🧠

## Model Chosen
- **Neural Network** with TensorFlow

### Why?
- Offers higher accuracy and better performance with complex feature interactions. 📈
- Provides a scalable solution for future feature additions. 🔧

## Usability
- **Rating**: 10.00 (Highly Usable)
- The application is user-friendly, providing both quick insights and detailed analysis options. 👌

## License
- **License**: Apache 2.0

## Expected Update Frequency
- Updates will be made as new data becomes available or when there are advances in machine learning techniques relevant to mental health prediction. 🔄

## Tags
- Mental Health 🧠
- Depression Prediction ☹️
- Student Well-being 🌱
- Machine Learning 🧑‍💻
- AI in Healthcare 🏥

---

## Instructions to Run Locally

#### Clone the project
```bash
  git clone https://github.com/Divyansh0108/MentalHealth/
```

#### Go to the project directory 

```bash
  cd MentalHealth
```

#### Create new environment and activate it

```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### Install dependencies

```bash
  pip install -r requirements.txt
```

#### Testing the Model:

```bash
  python scripts/model_testing.py
```

#### Running the Application Locally:

Before running, ensure you replace the API key:

Edit the "secrets.toml" file inside the .streamlit folder with your Google Generative AI API key:

```bash
API_KEY = "your_api_key_here"
```

#### Then finally run the app:

```bash
streamlit run scripts/app.py
```

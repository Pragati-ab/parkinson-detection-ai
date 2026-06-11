# Parkinson.AI — Voice-Based Parkinson's Prediction System

This project is a machine learning system that predicts Parkinson's disease risk based on voice features. The system was trained on the Oxford Parkinson's disease detection dataset and uses XGBoost for classification.

Dataset used: https://www.kaggle.com/datasets/thecansin/parkinsons-data-set 

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Training](#model-training)
5. [Tech Stack](#tech-stack)
6. [Results](#results)
7. [Deployment](#deployment)


## Overview
Parkinson.AI is a machine learning-based system that predicts the risk of Parkinson’s disease by analyzing voice features. Using the UCI Parkinson’s dataset, we extracted 22 acoustic features and built a predictive model using XGBoost.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/parkinson-ai.git
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install dependencies:
   ```bash
   pip install -r requirements.txt


## Usage
1. Start the backend API (FastAPI):
   ```bash
   uvicorn app:app --reload --port 8000
2. Access the frontend on your local machine (e.g., via Streamlit or a static HTML page).
3. Use the form to input voice measurements and click "Predict" to get results.


## Model Training
The dataset was split into training and test sets using an 80-20 split. We evaluated four models—Logistic Regression, SVM, Random Forest, and XGBoost—using 5-fold cross-validation. XGBoost consistently achieved the highest AUC and F1-score, prompting us to select it. Hyperparameter tuning was performed using grid search over learning rate, max depth, and regularization. The best model was saved using Joblib for deployment.


## Tech Stack
- Python
- XGBoost, Scikit-learn
- FastAPI (for backend)
- Pandas, NumPy (data manipulation)
- HTML, CSS, JavaScript (frontend)
- Jinja2 (templating)
- Joblib (model persistence)
- Render (for deployment)

## Results
The XGBoost model achieved an accuracy of 94%, an AUC of 0.96, and an F1-score of 0.93 on the test set. These results indicate a strong ability to differentiate between healthy individuals and those

![Alt text](../images/target_distribution.png)  
*Figure 1: Target distribution in dataset(0-> Healthy 1->Parkinson).*

![Alt text](../images/heatmap.png)  
*Figure 2: Heatmap.*

![Alt text](../images/model_comparison.png)  
*Figure 3: Model Comparison.*

![Alt text](../images/confusion_matrix.png)  
*Figure 1: Confusion Matrix of XGBoost.*

## Deployment
The model was deployed using FastAPI for the backend API, and the frontend was built as a static web page. The entire application is hosted on Render, enabling users to interact with the prediction API in real time.
     

# Stroke Prediction System

The **Stroke Prediction System** is a machine learning–based healthcare application designed to predict the likelihood of a stroke based on patient health data.  
It uses a trained ML model and provides predictions through a simple **Python application**, making it useful for early risk assessment and awareness.

---

## Overview

Stroke is one of the leading causes of death and long-term disability worldwide.  
Early identification of stroke risk can help in prevention and timely medical intervention.

This project uses historical healthcare data to train a machine learning model that predicts stroke occurrence based on factors such as age, gender, hypertension, heart disease, glucose levels, BMI, and lifestyle habits.

---

## Features

- Data cleaning and preprocessing pipeline
- Machine learning model training
- Saved trained model (`model.pkl`)
- Stroke risk prediction using user input
- Simple Python-based application interface
- Uses real healthcare stroke dataset

---

## Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Model:** Supervised Machine Learning classifier  
- **Data Format:** CSV  
- **Model Storage:** Pickle (`.pkl`)  

---

## Project Structure

<img width="697" height="261" alt="image" src="https://github.com/user-attachments/assets/6b00bfb9-4614-48b1-ae52-e95c8610856c" />

---

## Dataset

The dataset contains healthcare-related attributes such as:

- Age
- Gender
- Hypertension
- Heart disease
- Average glucose level
- BMI
- Smoking status
- Stroke occurrence (target variable)

The raw dataset is cleaned and processed before training the model.

---

## Model Training

1. Raw data is cleaned using `clean_data.py`
2. Missing values and categorical features are handled
3. Data is split into training and testing sets
4. Machine learning model is trained
5. Trained model is saved as `model.pkl`

---

## How the System Works

1. User provides health-related inputs
2. Input data is preprocessed
3. Trained model (`model.pkl`) is loaded
4. Model predicts stroke risk
5. Prediction result is displayed

---

## How to Run the Project

1. Clone the repository  
2. Install required Python libraries  
3. Run the application  

```bash
python app.py

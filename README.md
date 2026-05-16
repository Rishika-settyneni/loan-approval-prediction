# Loan Approval Prediction System

This project is a machine learning based web application that predicts whether a loan will be approved or not based on applicant details. The trained model is integrated with a Flask web application to provide predictions through a simple user interface.

## Project Overview

The goal of this project is to automate the loan approval decision process using historical loan application data. The model analyzes applicant information such as income, loan amount, credit history, and employment status to predict loan approval.

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML and CSS

## Machine Learning Model

- Trained using historical loan application data
- Data preprocessing includes handling missing values and encoding categorical features
- Model is saved using Pickle and loaded during prediction
- Predicts loan approval status based on user input

## Project Structure

loan-approval-prediction/
│── app.py
│── loan_approval.py
│── loan_model.pkl
│── Dataset.csv
│── static/
│ └── style.css
│── templates/
│ └── index.html
│── requirements.txt
│── README.md

graphql
Copy code

## How to Run the Project

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/loan-approval-prediction.git
cd loan-approval-prediction

markdown
Copy code

2. Install the dependencies

pip install -r requirements.txt

markdown
Copy code

3. Run the application

python app.py

css
Copy code

4. Open the application in a browser

http://127.0.0.1:5000/

markdown
Copy code

## Dataset

- The dataset contains historical loan application records
- Includes applicant financial and personal attributes
- Used to train the machine learning model

## Future Enhancements

- Improve prediction accuracy
- Add more input features
- Deploy the application online
- Improve user interface

## Author

Rishika Settyneni  
CSE (Artificial Intelligence and Machine Learning)

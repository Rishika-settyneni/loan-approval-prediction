from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained ML model
model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = 1 if request.form['Gender'] == 'Male' else 0
        married = 1 if request.form['Married'] == 'Yes' else 0
        dependents = 3 if request.form['Dependents'] == '3+' else int(request.form['Dependents'])
        education = 1 if request.form['Education'] == 'Graduate' else 0
        self_employed = 1 if request.form['Self_Employed'] == 'Yes' else 0
        applicant_income = float(request.form['ApplicantIncome'].replace(',', ''))
        coapplicant_income = float(request.form['CoapplicantIncome'].replace(',', ''))
        loan_amount = float(request.form['LoanAmount'].replace(',', ''))
        loan_term = float(request.form['Loan_Amount_Term'])
        credit_history = int(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        # Encode property area (2 dummies — one dropped)
        if property_area == 'Rural':
           prop_area = 0
        elif property_area == 'Semiurban':
            prop_area = 1
        else:  # Urban
            prop_area = 2


        # Combine features
        final_features = np.array([[gender, married, dependents, education,
                            self_employed, applicant_income, coapplicant_income,
                            loan_amount, loan_term, credit_history, prop_area]])

        prediction = model.predict(final_features)[0]
        output = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"
        print("Prediction result:", output)
        return render_template('index.html', prediction_text=output)

        return render_template('index.html', prediction_text=output)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)

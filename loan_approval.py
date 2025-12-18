# loan_approval.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Dataset.csv")

# Encode categorical columns
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']:
    data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_

# Predict
y_pred = best_model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Best Parameters:", grid.best_params_)
print("Accuracy:", round(acc * 100, 2), "%")
import pickle
pickle.dump(best_model, open('loan_model.pkl', 'wb'))

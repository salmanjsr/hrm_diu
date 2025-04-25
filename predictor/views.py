from django.shortcuts import render
import pickle
import numpy as np
import os
import pandas as pd

# Base path & CSV path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, 'predictor', 'Perfomance_report.csv')

# Load CSV
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"❌ CSV load error: {e}")

# Load model & scaler
model = pickle.load(open(os.path.join(BASE_DIR, 'predictor', 'model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'predictor', 'scaler.pkl'), 'rb'))

# New function to evaluate employee based on performance and satisfaction
def evaluate_employee(employee):
    if employee['EmpJobSatisfaction'] >= 4 and employee['PerformanceRating'] >= 4:
        return "Continue"
    elif employee['EmpJobSatisfaction'] == 3 and employee['PerformanceRating'] == 3:
        return "Review"
    elif employee['EmpJobSatisfaction'] <= 2 and employee['PerformanceRating'] <= 2:
        return "Terminate"
    else:
        return "Review"

def home(request):
    prediction = None
    employee_data = None
    decision = None  # Initialize decision variable

    if request.method == 'POST':
        try:
            age = float(request.POST.get('age'))
            work_hours = float(request.POST.get('work_hours'))  # EmpHourlyRate
            satisfaction = float(request.POST.get('satisfaction'))  # EmpJobSatisfaction

            features = np.array([[age, work_hours, satisfaction]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            # Match employee info
            match = df[
                (df['Age'] == age) &
                (df['EmpHourlyRate'] == work_hours) &
                (df['EmpJobSatisfaction'] == satisfaction)
            ]

            if not match.empty:
                # Evaluate employee decision
                match['Decision'] = match.apply(evaluate_employee, axis=1)
                employee_data = match.to_html(classes='table table-bordered', index=False)
                decision = match['Decision'].iloc[0]  # Get decision for the matched employee
            else:
                employee_data = "<p class='text-danger'>No matching employee found.</p>"

        except Exception as e:
            prediction = f"Error: {e}"

    return render(request, 'index.html', {
        'prediction': prediction,
        'sample_data': employee_data,
        'decision': decision  # Add decision to the context
    })

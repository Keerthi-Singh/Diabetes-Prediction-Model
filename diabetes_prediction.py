import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (no need for names=column_names as they're already in the file)
url = r"C:\Users\keert\Downloads\archive\diabetes.csv"  # Update with your local path
df = pd.read_csv(url)

# Ensure dataset has the right columns (check the first few rows)
print(df.head())

# Prepare data for training
X = df.drop('Outcome', axis=1)  # Features (everything except the 'Outcome' column)
y = df['Outcome']  # Target (the 'Outcome' column)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")


# Function to take user input and predict diabetes
def predict_diabetes():
    print("Please enter the following details to predict if you have diabetes:")

    # Ensure inputs are numeric
    try:
        pregnancies = int(input("Number of Pregnancies: "))
        glucose = int(input("Glucose level: "))
        blood_pressure = int(input("Blood Pressure (mm Hg): "))
        skin_thickness = int(input("Skin Thickness (mm): "))
        insulin = int(input("Insulin level (mu U/ml): "))
        bmi = float(input("BMI (Body Mass Index): "))
        pedigree_function = float(input("Diabetes Pedigree Function: "))
        age = int(input("Age: "))
    except ValueError:
        print("Invalid input, please enter numeric values only.")
        return  # Exit if invalid input is entered

    # Prepare the input data for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Output the result
    if prediction == 1:
        print("The model predicts that you have diabetes.")
    else:
        print("The model predicts that you do not have diabetes.")

# Call the function to get user input and predict
predict_diabetes()

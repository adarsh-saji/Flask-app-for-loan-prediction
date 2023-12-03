

from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and encoders outside of the route functions
with open('Random_Forest.pkl', 'rb') as file:
    rf2_final = pickle.load(file)

    feature_names = [
        'Client_Income', 'Active_Loan', 'House_Own', 'Credit_Amount', 'Loan_Annuity',
        'Client_Income_Type', 'Client_Education', 'Client_Marital_Status', 'Client_Gender',
        'Loan_Contract_Type', 'Age_Days', 'Employed_Days', 'Client_Occupation',
        'Type_Organization', 'Vehicle_Owned', 'Family_Members'
    ]
    rf2_final.feature_names = feature_names

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)
with open('ordinal_encoding.pkl', 'rb') as file:
    oe = pickle.load(file)
with open('robust_scaling.pkl', 'rb') as file:
    robust_scaling = pickle.load(file)

# Define the columns used in preprocessing
cols_to_scale = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 'Employed_Days']
ordinal_cols = ['Client_Education']
label_cols = ['Client_Income_Type', 'Client_Marital_Status', 'Client_Gender',
              'Client_Occupation', 'Type_Organization', 'Loan_Contract_Type']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    # Create a dictionary to store user input data
    input_data = {
        'Client_Income': float(request.form.get('Client_Income')),
        'Active_Loan': float(request.form.get('Active_Loan')),
        'House_Own': float(request.form.get('House_Own')),
        'Credit_Amount': float(request.form.get('Credit_Amount')),
        'Loan_Annuity': float(request.form.get('Loan_Annuity')),
        'Client_Income_Type': request.form.get('Client_Income_Type'),
        'Client_Education': request.form.get('Client_Education'),
        'Client_Marital_Status': request.form.get('Client_Marital_Status'),
        'Client_Gender': request.form.get('Client_Gender'),
        'Loan_Contract_Type': request.form.get('Loan_Contract_Type'),
        'Age_Days': float(request.form.get('Age_Days')),
        'Employed_Days': float(request.form.get('Employed_Days')),
        'Client_Occupation': request.form.get('Client_Occupation'),
        'Type_Organization': request.form.get('Type_Organization'),
        'Vehicle_Owned': float(request.form.get('Vehicle_Owned')),
        'Family_Members': float(request.form.get('Family_Members'))
    }

    # Create a DataFrame from the input data
    input_data_df = pd.DataFrame([input_data])
    
    # Ordinal encoding
    encoded_ordinal = oe.transform(input_data_df[ordinal_cols])
    
    # Label encoding
    encoded_label_cols = []
    for col, le in label_encoders.items():
        encoded_col = le.transform(input_data_df[col])
        encoded_label_cols.append(encoded_col)

    encoded_label = np.column_stack(encoded_label_cols)

      
    # Robust scaling for specified columns
    scaled_cols = []
    for col, rs in robust_scaling.items():
        scaled_col = rs.transform(input_data_df[[col]])
        scaled_cols.append(scaled_col)

    scaled = np.column_stack(scaled_cols)

    
    # Extract non-scaled and non-encoded columns from input_data_df
    non_scaled_encoded_cols = [col for col in input_data_df.columns if col not in cols_to_scale + ordinal_cols + label_cols]

    # Extract the corresponding data from input_data_df
    non_scaled_encoded_data = input_data_df[non_scaled_encoded_cols].values

    # Concatenate all parts together
    encoded_input = np.concatenate((encoded_ordinal, encoded_label, scaled, non_scaled_encoded_data), axis=1)
    # Create a DataFrame with the encoded input data and feature names
    encoded_input_df = pd.DataFrame(encoded_input, columns=feature_names)
    print(encoded_input_df.head())
    

    # Make the prediction
    prediction = rf2_final.predict(encoded_input_df)

    # Verify the class labels
    class_labels = rf2_final.classes_

    predicted_result = class_labels[prediction][0] 
    print(predicted_result)

    # Map the predicted class label to the corresponding label
    if predicted_result == 0:  # Check for Non-Default class label
        predicted_result= "Non-Default"
    else:  # Check for Default class label
        predicted_result = "Default"

    # Render the prediction result
    return render_template('result.html', prediction_text='Predicted Loan Default: {}'.format(predicted_result))

if __name__ == '__main__':
    app.run(debug=False)

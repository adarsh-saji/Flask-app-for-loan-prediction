

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("cleaned2.csv")

# Columns for ordinal encoding
ordinal_cols = ['Client_Education']

# Columns for label encoding
label_cols = ['Client_Income_Type', 'Client_Marital_Status', 'Client_Gender',
              'Client_Occupation', 'Type_Organization', 'Loan_Contract_Type']

# Create an instance of the OrdinalEncoder
oe = OrdinalEncoder()

# Encode columns using OrdinalEncoder
encoded_ordinal = oe.fit_transform(df[ordinal_cols])

# Replace the original columns with the encoded values
df[ordinal_cols] = encoded_ordinal

# Encode columns using LabelEncoder

label_encoders = {}  # To store encoders for each column

for col in label_cols:
    le = LabelEncoder()
    encoded = le.fit_transform(df[col])
    df[col] = encoded
    label_encoders[col] = le

# To handle the outliers we are using the capping technique
Q1 = np.percentile(df['Employed_Days'], 25, interpolation='midpoint')
Q3 = np.percentile(df['Employed_Days'], 75, interpolation='midpoint')
IQR = Q3 - Q1

up_lim = Q3 + 1.5 * IQR
low_lim = Q1 - 1.5 * IQR

df['Employed_Days'] = np.where(df['Employed_Days'] > up_lim, up_lim,
                               np.where(df['Employed_Days'] < low_lim, low_lim, df['Employed_Days']))

# Scaling
cols_to_scale = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Age_Days', 'Employed_Days']

robust_scaling = {}  # To store scaling for each column

for col in cols_to_scale:
    rs = RobustScaler()
    scaled_values = rs.fit_transform(df[[col]])  # Notice the change here, using rs.fit_transform
    df[col] = scaled_values
    robust_scaling[col] = rs


# Splitting Data
x = df.drop("Default", axis=1)
y = df["Default"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.40)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=42, test_size=0.50)

# Assign feature names to the model
feature_names = [
    'Client_Income', 'Active_Loan', 'House_Own', 'Credit_Amount', 'Loan_Annuity',
    'Client_Income_Type', 'Client_Education', 'Client_Marital_Status', 'Client_Gender',
    'Loan_Contract_Type', 'Age_Days', 'Employed_Days', 'Client_Occupation',
    'Type_Organization', 'Vehicle_Owned', 'Family_Members'
]

# Hyperparameter Tuning
param_dist = {'n_estimators': [10, 50, 100, 200, 500],
              'max_depth': [5, 10, 50, 100, None],
              'criterion': ["gini", "entropy"]}

rf = RandomForestClassifier()

rf_random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)
rf_random_search.fit(x_train, y_train)


# Training and Evaluating Models
rf2_final = RandomForestClassifier(n_estimators=500, max_depth=50, criterion='entropy')
rf2_final.fit(x_train, y_train)



rf2_final.feature_names = feature_names

# Save the trained model and encoders using pickle
with open('Random_Forest.pkl', 'wb') as file:
    pickle.dump(rf2_final, file)

with open('label_encoders.pkl', 'wb') as file:
    pickle.dump((label_encoders), file)

with open('ordinal_encoding.pkl', 'wb') as file:
    pickle.dump(oe, file)

with open('robust_scaling.pkl', 'wb') as file:
    pickle.dump(robust_scaling, file)

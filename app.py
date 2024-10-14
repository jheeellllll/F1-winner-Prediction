# Required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import pickle
import streamlit as st

# Load the datasets
races_df = pd.read_csv('races.csv')
results_df = pd.read_csv('results.csv')

# Merge the datasets on 'season', 'round', and 'circuit_id'
merged_df = results_df.merge(races_df, on=['season', 'round', 'circuit_id'])

# Feature engineering
# Create a binary target variable 'podium' where 1 represents a win (1st place), and 0 represents other positions
merged_df['podium'] = merged_df['podium'].apply(lambda x: 1 if x == 1 else 0)

# Select relevant features for prediction
# We use 'grid', 'season', 'round', 'constructor', 'driver', and 'qualifying_time' as predictive features
features = ['grid', 'season', 'round', 'constructor', 'driver', 'points']
X = merged_df[features]
y = merged_df['podium']

# Encode categorical variables such as 'constructor' and 'driver' to numerical values
X = pd.get_dummies(X, columns=['constructor', 'driver'])

# Save the column names before imputation
columns = X.columns

# Handle missing values by imputing the median (common in real-world datasets)
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for better model convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network Classifier setup (Multi-Layer Perceptron with 4 hidden layers)
model = MLPClassifier(hidden_layer_sizes=(80, 20, 40, 5), activation='relu', solver='adam', alpha=0.001, random_state=1)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save the model and pre-processing objects
with open('f1_model.pkl', 'wb') as model_file:
    pickle.dump((model, scaler, imputer, columns), model_file)

# Streamlit app for user interaction
st.title('F1 Race Winner Prediction')

# User inputs for race prediction
st.header('Input the race details:')
grid = st.number_input('Grid Position', min_value=1, max_value=20, value=1, step=1)
season = st.number_input('Season', min_value=1950, max_value=2024, value=2024, step=1)
round_num = st.number_input('Round Number', min_value=1, max_value=23, value=1, step=1)
constructor = st.selectbox('Constructor', sorted(merged_df['constructor'].unique()))
driver = st.selectbox('Driver', sorted(merged_df['driver'].unique()))
points = st.number_input('Driver\'s Current Points', min_value=0, max_value=100, value=0, step=1)

# When the user clicks the "Predict" button
if st.button('Predict'):
    # Prepare user input for model prediction
    new_race_data = [[grid, season, round_num, constructor, driver, points]]
    new_race_df = pd.DataFrame(new_race_data, columns=['grid', 'season', 'round', 'constructor', 'driver', 'points'])
    
    # Convert input data to match the dummy variables used during training
    new_race_df = pd.get_dummies(new_race_df).reindex(columns=columns, fill_value=0)  # Align columns with the training data
    new_race_scaled = scaler.transform(imputer.transform(new_race_df))  # Apply imputation and scaling

    # Make a prediction
    winner_prediction = model.predict(new_race_scaled)
    
    # Display the prediction result
    if winner_prediction[0] == 1:
        st.success(f"The predicted winner is {driver}!")
    else:
        st.error(f"{driver} is not predicted to win the race.")

# Display model evaluation results
st.header('Model Evaluation')
st.write(f'Accuracy: {accuracy:.2f}')
st.text(report)

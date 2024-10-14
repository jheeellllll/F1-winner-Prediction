# F1-winner-Prediction
This project predicts Formula 1 race winners using historical race data and machine learning. The model is built using a neural network (MLPClassifier) and considers features like grid position, season, round, constructor, driver, and points.

The repository includes a frontend interface using Streamlit for user interaction, allowing predictions based on real-world race scenarios.
Overview
This project leverages machine learning to predict Formula 1 race winners based on historical race data. It uses a Multi-Layer Perceptron (MLPClassifier) neural network model to predict the likelihood of a driver winning a race based on features like grid position, season, round, constructor, and driver. The project also includes a Streamlit frontend, allowing users to input race details and see predictions in real time.

Features
Predict the winner of a Formula 1 race using historical data.
Machine learning model built with sklearn's MLPClassifier.
Streamlit web app for interactive predictions.
Considers multiple important race factors like grid position, driver, constructor, and points.
Model evaluation included with accuracy and classification reports.

Dataset
The project uses two primary datasets:

races.csv – Contains race-specific information (season, round, circuit, date, etc.).

results.csv – Contains race results with driver, constructor, grid position, points, etc.

These datasets are merged on season, round, and circuit_id to create a comprehensive dataset for modeling.

Model
The model used in this project is a neural network classifier (MLPClassifier) with the following configuration:

Hidden Layers: 4 layers with sizes (80, 20, 40, 5)

Activation: ReLU

Solver: Adam optimizer

Alpha: 0.001 (L2 regularization)

The model is trained to predict a binary outcome:

1 if the driver is predicted to win the race.
0 otherwise.

Installation
To run the project locally, follow these steps:

Clone the repository:

bash
git clone https://github.com/your-username/f1-race-winner-prediction.git
cd f1-race-winner-prediction
Install dependencies: It's recommended to use a virtual environment:

bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
Run the Streamlit app:

bash
streamlit run app.py
Load the model and datasets: Make sure the datasets (races.csv and results.csv) are in the project folder and correctly formatted.

Usage
Once the app is running, you can:

Input race details like grid position, season, round, driver, constructor, and current points.
Click the "Predict" button to see if the selected driver is predicted to win the race.
View model performance metrics including accuracy and classification report.

Model Evaluation
The model achieves reasonable accuracy in predicting race winners, with the following metrics based on the test data:

Accuracy: ~X%
Precision, Recall, F1-score: Detailed in the classification report provided in the app.
Future Improvements
Integrate additional features like weather conditions, driver form, or track type.
Experiment with different machine learning models (e.g., Random Forest, Gradient Boosting).
Enhance the user interface to allow for more detailed race customization.

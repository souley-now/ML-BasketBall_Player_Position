import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
import pickle

# Load data
data = pd.read_csv('nba-player-stats-2021.csv')

# Handle missing values and duplicates
data = data.dropna().drop_duplicates()

# Feature selection
X = data.drop(columns=['player', 'pos', 'tm'])
y = data['pos']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Initialize Flask app
app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    features = data['features']
    
    # Ensure the features are in the correct format
    features = [features]
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
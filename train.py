# Import necessary libraries
import numpy as np  # Add this line to import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

# Import the data extraction function from getdata.py
from getdata import extract_features_labels

# Load the data
features, labels = extract_features_labels('annotations.json')

# Choose a machine learning model
model = RandomForestClassifier()

print("model defined")

# Train the model
model.fit(features, labels)

# Evaluate the model
scores = cross_val_score(model, features, labels, cv=5)
print("Cross-validated Accuracy:", np.mean(scores))

# Save the trained model
joblib.dump(model, 'jiu_jitsu_pose_model.pkl')

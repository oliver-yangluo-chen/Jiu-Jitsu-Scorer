
import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

from getdata import extract_features_labels

# Load  data
features, labels = extract_features_labels('annotations.json')

model = RandomForestClassifier()

print("model defined")

# Train
model.fit(features, labels)

# Evaluate 
scores = cross_val_score(model, features, labels, cv=5)
print("Cross-validated Accuracy:", np.mean(scores))

# Save 
joblib.dump(model, 'jiu_jitsu_pose_model.pkl')

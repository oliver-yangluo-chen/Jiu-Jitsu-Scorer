import joblib
import json


vid_path = "bjjvjudo"


def getpredictions():
    # Load the trained model
    model = joblib.load('jiu_jitsu_pose_model.pkl')
    # Make predictions on the new data
    with open('bteam_features1.json', 'r') as file:
        features = json.load(file)
        new_predictions = model.predict(list(features.values()))
        return new_predictions.tolist()

predictions = getpredictions()

with open('predictions.json', 'w') as file:
    json.dump(predictions, file)

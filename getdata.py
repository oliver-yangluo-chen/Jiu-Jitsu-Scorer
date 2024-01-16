import json
import numpy as np

def extract_features_labels(json_file):
    """
    Extract features and labels from the given JSON file.
    
    :param json_file: Path to the JSON file containing the frame data.
    :return: Tuple of (features, labels), where features is a list of joint coordinates
             and labels is a list of Jiu-Jitsu positions.
    """
    with open(json_file, 'r') as file:
        frames = json.load(file)

    features = []
    labels = []

    for frame in frames:
        # Check if both 'pose1' and 'pose2' are present
        if 'pose1' in frame and 'pose2' in frame:
            position = frame['position']
            pose1 = frame['pose1']
            pose2 = frame['pose2']

            # Ensure there are 17 joints in each pose
            assert len(pose1) == 17 and len(pose2) == 17, "Each pose must have 17 joints."

            # Combining pose1 and pose2 into a single feature vector
            combined_pose = []
            for joint in pose1 + pose2:
                combined_pose.extend(joint[:2])  # Append only x and y coordinates

            features.append(combined_pose)
            labels.append(position)

    return np.array(features), np.array(labels)

from gen_keypts import gen_keypts
import joblib
import os
import json

# Load new data

def writefeatures(vid_path):
    img_paths = os.listdir(vid_path)
    img_paths = sorted(img_paths)
    print(img_paths)

    with open('bteam_features1.json', 'w') as file:
        json.dump({}, file)

    for i in img_paths:
        feature = gen_keypts(vid_path + '/' + i)
        if len(feature) == 68: 
            with open('bteam_features1.json', 'r') as file:
                data = json.load(file)
            data[i] = feature

            with open('bteam_features1.json', 'w') as file:
                json.dump(data, file)
    return None

writefeatures("bteam1")




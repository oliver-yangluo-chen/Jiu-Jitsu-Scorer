import json

with open('bteam_features1.json', 'r') as file:
    features = json.load(file)

with open('predictions.json', 'r') as file:
    predictions = json.load(file)

zipped = dict(zip(features.keys(), predictions))
print(zipped)


from PIL import Image, ImageDraw, ImageFont
import cv2
import os


last_pose = ''

vid_path = "bteam_out1"
img_paths = os.listdir(vid_path)
img_paths = sorted(img_paths)
print(img_paths)

for i in img_paths[1:]:
    img = cv2.imread(vid_path + '/' + i)
    draw_img = img.copy()

    text = zipped.get('0'+i, last_pose)
    last_pose = text
    position = (50, 50)  # Change as needed
    font = cv2.FONT_HERSHEY_SIMPLEX  # You can choose different fonts
    font_scale = 1
    font_color = (0, 0, 0)  # White color
    line_type = 2

    cv2.putText(draw_img, text, position, font, font_scale, font_color, line_type)

    cv2.imwrite("bteam_out1/"+i, draw_img)


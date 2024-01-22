import cv2
import os
import glob

# Path to the folder containing png images
vid_path = 'bteam_out1'

# Retrieve a list of the image file paths
img_paths = os.listdir(vid_path)
img_paths = sorted(img_paths)[1:]

# Determine the width and height from the first image
image = cv2.imread(vid_path+'/'+img_paths[0])
height, width, layers = image.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'x264' if you prefer

# Set the desired FPS
fps = 15  # For example, 30 FPS

# Create VideoWriter object
video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

for file in img_paths:
    image = cv2.imread(vid_path+'/'+file)
    video.write(image)

# Release the video writer
video.release()

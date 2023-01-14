import cv2
import numpy as np

img_size = (256, 256)

def create_video(images, width, height):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('sample.mp4', fourcc, 20, (width, height))
    for i in range(len(images)):
        img = images[i]
        img_np = np.array(img).astype('uint8')
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        writer.write(img_cv)
    writer.release()
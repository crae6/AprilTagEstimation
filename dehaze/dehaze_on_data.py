import os,argparse
from matplotlib import scale
import numpy as np
from PIL import Image
from FFA import *
import torch
import cv2
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dcp import *
from retinex import *
from pupil_apriltags import Detector
from dehaze import *
import csv
import time
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

BASELINE = 0
DCP = 1
SINGLE_SCALE = 2
MULTI_SCALE = 3
MODEL = 4

names = ['baseline','dcp','single_scale','multi_scale','model']

ISOLATE_HAZE = False  # change this to True to only run on one image for debugging

DEHAZE_METHOD = BASELINE  # change this to switch dehazing methods

def processed2gray(img):
    gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    gray = gray.astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return gray 


# fx = 1.06602296e+03
# fy = 1.05918899e+03
# cx = 4.01884586e+02
# cy = 2.90936676e+02

fx = 2.48502856e+03
fy = 2.48095083e+03
cx = 1.67655746e+03
cy = 1.33820137e+03

camera_params = (fx, fy, cx, cy)
data_path = '../dataset/'
if __name__ == '__main__':
    csv_path = "../dataset/data.csv"
    detections = []
    depths = {}
    num_images = 0
    if DEHAZE_METHOD == MODEL:
        net = init_model()

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        start = time.time()
        for row in reader:
            image_name = row[0]      # string
            depth = float(row[1])    # convert from text to number
            image_path = os.path.join(data_path, image_name)
            if ISOLATE_HAZE:
                if 'haze' not in image_name:
                    continue
            print('Processing image: ', image_name)
            
            num_images += 1

            # pi_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            pi_image = cv2.imread(image_path) # BGR

            if DEHAZE_METHOD == BASELINE:
                pi_image = cv2.cvtColor(pi_image, cv2.COLOR_BGR2GRAY)
                processed_img = pi_image

            elif DEHAZE_METHOD == DCP:
                processed_img = dcp(pi_image)
                processed_img = processed2gray(processed_img)

            elif DEHAZE_METHOD == SINGLE_SCALE:
                # gray = cv2.cvtColor(pi_image, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
                processed_img = SSR(pi_image, variance=30)
                processed_img = processed2gray(processed_img)

            elif DEHAZE_METHOD == MULTI_SCALE:
                # processed_img = MSR(pi_image, variance_list=[15, 60, 120])
                processed_img = MSR(pi_image, variance_list=[15, 80])
                processed_img = processed2gray(processed_img)

            elif DEHAZE_METHOD == MODEL:
                ffa_img = cv2.resize(pi_image, None, fx=0.25, fy=0.25)
                dehaze, dehaze_cv = ffa_dehaze(ffa_img, net)
                processed_img = processed2gray(dehaze_cv)
            #convert to grayscale for april tag detection
            #  = cv2.cvtColor(pi_image, cv2.COLOR_BGR2GRAY)

            rpi_detect = detector.detect(
                processed_img,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=0.19
            )
            if len(rpi_detect):
                detection_depth = rpi_detect[0].pose_t[2][0]
                depths[image_name] = detection_depth
                detections.append(image_name)
            
            
            # if len(rpi_detect):
            #     cv2.imshow('base',pi_image)
            #     cv2.imshow('dcp',pi_image_dcp)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

    end = time.time()
    
    print(f"Processed {num_images} images in {end - start:.2f} seconds")
    print(f'Number of detections: {len(detections)}')
    np.save(f'depths_{names[DEHAZE_METHOD]}.npy', depths)
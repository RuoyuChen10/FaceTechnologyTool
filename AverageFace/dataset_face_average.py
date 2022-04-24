# -*- coding: utf-8 -*-  

"""
Created on 2022/4/16

@author: Ruoyu Chen
"""

# Standard imports
import os
import argparse
import cv2
import numpy as np

from face_landmarks import detect_landmarks
from face_average import AverageFace

import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Average Face for Dataset')
    # general
    parser.add_argument('--image-dir', type=str, default="results/VGGFace2-Male",
                        help='input image path')
    parser.add_argument('--size',
                        type=int,
                        default=224,
                        help='size of the generated average face image')
    parser.add_argument('--save-dir',
                        type=str,
                        default='./results/VGGFace-male.png',
                        help='save dir.')
    args = parser.parse_args()
    return args

# Read points from text files in directory
def readPoints(path) :
    detect_landmarks(path)

    # Create an array of array of points.
    pointsArray = []

    #List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        
        if filePath.endswith(".txt"):
            
            #Create an array of points.
            points = []            
            
            # Read points from filePath
            with open(os.path.join(path, filePath)) as file :
                for line in file :
                    x, y = line.split()
                    points.append((int(x), int(y)))
            
            # Store array of points
            pointsArray.append(points)
            
    return pointsArray

# Read all jpg images in folder.
def readImages(path) :

    #Create array of array of images.
    imagesArray = []

    im_paths = glob.glob(os.path.join(path, '*.jpg'))
    im_paths += glob.glob(os.path.join(path, '*.jpeg'))
    im_paths += glob.glob(os.path.join(path, '*.png'))

    print(im_paths)

    #List all files in the directory and read points from text files one by one
    for filePath in sorted(im_paths):
        # Read image found.
        img = cv2.imread(filePath)

        # Convert to floating point
        img = np.float32(img)/255.0

        # Add to array of images
        imagesArray.append(img)

    return imagesArray

def main(args):
    face_aver = AverageFace(args.size,args.size)
    facial_landmarks = readPoints(args.image_dir)
    images = readImages(args.image_dir)

    # Save result
    cv2.imwrite(args.save_dir, face_aver.face_average(images,facial_landmarks) * 255)

if __name__ == "__main__":
    args = parse_args()
    main(args)
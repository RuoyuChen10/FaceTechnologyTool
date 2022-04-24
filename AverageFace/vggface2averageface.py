# -*- coding: utf-8 -*-  

"""
Created on 2021/12/11

@author: Ruoyu Chen
Generate an average face from vggface2 dataset
"""

# Standard imports
import os
import argparse
import cv2
import numpy as np

from face_average import AverageFace

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--image-dir', type=str, default="/home/cry/data2/VGGFace2/train/",
                        help='input image path')
    parser.add_argument('--landmark-dir',
                        type=str,
                        default='./VGGFace2',
                        help='landmark_dir.')
    parser.add_argument('--process', type=int, default=0,
                        help='the processing')
    parser.add_argument('--size',
                        type=int,
                        default=224,
                        help='size of the generated average face image')
    parser.add_argument('--save-dir',
                        type=str,
                        default='./results/VGGFace2',
                        help='save dir.')
    args = parser.parse_args()
    return args

def single_person(image_dir, landmark_dir):
    """
    Calculate single people average face 

    image_dir: /home/cry/data2/VGGFace2/train/n000138
    landmark_dir: VGGFace2/n000138
    """
    # store image
    imagesArray = []
    # store landmark
    pointsArray = []
    
    people_lds = os.listdir(landmark_dir)

    for people_ld in people_lds:
        try:
            # landmark txt path
            people_ld_path = os.path.join(landmark_dir, people_ld)
            # correspond img path
            people_img_path = os.path.join(image_dir, people_ld.replace(".txt", ".jpg"))

            #Create an array of points.
            points = []            
            
            # Read points from filePath
            with open(people_ld_path) as file :
                for line in file :
                    x, y = line.split()
                    points.append((int(x), int(y)))

            # read image
            img = cv2.imread(people_img_path)
            # Convert to floating point
            img = np.float32(img)/255.0

            # Store array of points
            pointsArray.append(points)
            # Add to array of images
            imagesArray.append(img)
        except:
            pass

    return imagesArray, pointsArray

    # cv2.imwrite('results/{}.png'.format("example_average_face"), face_aver.face_average(images,facial_landmarks) * 255)

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def main(args):
    mkdir(args.save_dir)

    img_path = args.image_dir
    ldmk_dir  = args.landmark_dir

    # Average face
    face_aver = AverageFace(args.size, args.size)

    people_id = os.listdir(ldmk_dir)
    people_id = people_id[args.process:]
    
    for people in tqdm(people_id):  
        # path of the file
        people_ld_path = os.path.join(ldmk_dir, people)
        people_image_path = os.path.join(img_path, people)

        # get from each people
        imagesArray, pointsArray = single_person(people_image_path, people_ld_path)

        # save
        save_path = os.path.join(args.save_dir, people + ".jpg")
        try:
            cv2.imwrite(save_path, face_aver.face_average(imagesArray, pointsArray) * 255)
        except:
            print(people)

if __name__ == "__main__":
    args = parse_args()
    main(args)
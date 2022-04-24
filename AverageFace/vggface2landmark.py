# -*- coding: utf-8 -*-  

"""
Created on 2021/12/11

@author: Ruoyu Chen
Generate an average face from vggface2 dataset
"""

# Standard imports
import os
import glob
import urllib.request
import bz2
import argparse

# Dependecy imports
import dlib
from skimage import io

from tqdm import tqdm

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def download_if_not_exist():
    """Download predictor trained model if not exist to detect face."""

    predictor_path = 'shape_predictor_68_face_landmarks.dat'

    if not os.path.isfile(predictor_path):

        print('Downloading from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        urllib.request.urlretrieve(
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            "shape_predictor_68_face_landmarks.dat.bz2")
        print('Done')

        archive_path = 'shape_predictor_68_face_landmarks.dat.bz2'

        zipfile = bz2.BZ2File(archive_path) # open the file
        data = zipfile.read() # get the decompressed data
        open(predictor_path, 'wb').write(data) # write a uncompressed file
        del data

        # Remove archive
        os.remove(archive_path)

    return predictor_path

def detect_landmarks(dir_path, save_dir):
    """Detect facial landmarks using pretrained model."""

    predictor_path = download_if_not_exist()

    detector = dlib.get_frontal_face_detector() # pylint: disable=E1101
    predictor = dlib.shape_predictor(predictor_path) # pylint: disable=E1101
    # win = dlib.image_window()

    image_list = os.listdir(dir_path)

    for im_name in image_list:

        if (".jpg" in im_name) or (".png" in im_name):
            if not os.path.isfile(os.path.join(save_dir, im_name.replace('.jpg', '.txt'))):
                im_path = os.path.join(dir_path, im_name)
                # Processing image
                try:
                    img = io.imread(im_path)

                    # Ask the detector to find the bounding boxes of each face. The 1 in the
                    # second argument indicates that we should upsample the image 1 time. This
                    # will make everything bigger and allow us to detect more faces.
                    dets = detector(img, 1)
                    # print("Number of faces detected: {}, choosing biggest".format(len(dets)))
                    if len(dets) !=0:
                        print("not 0")
                        areas = []
                        for det in dets:
                            areas.append(det.area())

                        biggest_area_id = areas.index(max(areas))

                        # Get the landmarks/parts for the face in box d.
                        shape = predictor(img, dets[biggest_area_id])

                        with open(os.path.join(save_dir, im_name.replace('.jpg', '.txt')), "a") as myfile:
                            for i in range(shape.num_parts):
                                myfile.write(str(shape.part(i).x) + ' ' + str(shape.part(i).y) + '\n')
                except:
                    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--image-dir', type=str, default="/home/cry/data2/VGGFace2/train/",
                        help='input image path')
    parser.add_argument('--process', type=int, default=0,
                        help='the processing')
    parser.add_argument('--save-dir',
                        type=str,
                        default='./VGGFace2',
                        help='save dir.')
    args = parser.parse_args()
    return args

def main(args):
    root_path = args.image_dir
    save_dir  = args.save_dir

    people_id = os.listdir(root_path)
    people_id = people_id[args.process:]

    for people in tqdm(people_id):
        mkdir(os.path.join(save_dir, people))
        detect_landmarks(
            os.path.join(root_path, people), 
            os.path.join(save_dir, people)
            )

def one_people(args):
    """
    only detect one people
    """
    root_path = args.image_dir
    save_dir  = args.save_dir

    peoples = [
        "n001321",
        "n006732",
        "n009204",
        "n005767",
        "n005679",
        "n004758",
        "n006174",
        "n001134",
        "n006399",
        "n000361",
        "n002020",
        "n002777",
        "n001732",
        "n005153",
        "n001638",
        "n000946",
        "n001497",
        "n008142",
        "n007925",
        "n000125",
        "n004922",
        "n007526",
        "n004163",
        "n003952",
    ]

    # for people in tqdm(peoples):
    #     mkdir(os.path.join(save_dir, people))
    #     files = os.listdir(os.path.join(save_dir, people))

    #     for file in files:
    #         file_path = os.path.join(os.path.join(save_dir, people),file)
    #         os.remove(file_path)  
    people = "n007925"
    detect_landmarks(
        os.path.join(root_path, people), 
        os.path.join(save_dir, people)
        )

if __name__ == "__main__":
    args = parse_args()
    main(args)
    # one_people(args)
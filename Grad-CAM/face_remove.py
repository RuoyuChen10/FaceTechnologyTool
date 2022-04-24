import os 
import cv2
import numpy as np

from models.face_parser import FaceParser, remove

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def main():
    model = FaceParser()
    image_root = '/home/cry/data2/VGGFace2/train_align_arcface/'
    save_dir = "./mask_image_org"
    id_txt = "remove.txt"

    attribute = ['background', 'mouth', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface']
    attribute_ = ['mouth', 'eyebrows', 'eyes', 'hair', 'nose']

    f=open('image_list.txt')
    datas = f.readlines()  # 直接将文件中按行读到list里，效
    f.close()  # 关

    
main()


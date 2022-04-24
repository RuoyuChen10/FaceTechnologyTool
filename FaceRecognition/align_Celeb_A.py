# -*- coding: utf-8 -*-  

"""
Created on 2021/05/11

@author: Ruoyu Chen
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


from tools.alignment import FaceAlignment

landmark_list = "/exdata2/RuoyuChen/CelebA/Anno/list_landmarks_align_celeba.txt"
face_image_dir_path = "/exdata2/RuoyuChen/CelebA/Img/img_align_celeba"
save_dir_path = "/exdata2/RuoyuChen/CelebA/Img/img_align_crop_celeba"

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def main():
    mkdir(save_dir_path)

    with open(landmark_list,"r") as file:
        datas = file.readlines()

    # alignment
    face_align = FaceAlignment()

    for i in tqdm(range(2,len(datas))):
        if i < 67528:
            continue
        # alignment
        image_name = datas[i].split(" ")[0]
        image_path = os.path.join(face_image_dir_path, image_name)

        try:
            image = cv2.imread(image_path)
            
            strs = datas[i]
            strs = strs.splitlines()[0]
            strs = strs.split(" ")
            while "" in strs:
                strs.remove("")
            
            landmarks = np.array([int(data_) for data_ in strs[1:]])
            lnk = landmarks.reshape((-1,5,2)).astype(int)[0] # The first landmark
            warp_image = face_align(image, lnk)

            cv2.imwrite(os.path.join(save_dir_path, image_name), warp_image)
        except:
            print(image_path)

if __name__ == "__main__":
    main()
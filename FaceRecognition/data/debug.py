import cv2
import os
from tqdm import tqdm

f=open('seed1-1000/train.txt')
datas = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
f.close()  # 关

for data in tqdm(datas):
    image_name = data.split(" ")[0]
    image_path = os.path.join("/exdata2/RuoyuChen/Datasets/VGGFace2/train_align_arcface", image_name)

    image = cv2.imread(image_path)
    if image is None:
        print(image_name )
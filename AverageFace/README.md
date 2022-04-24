# Average Face

The function of this tool is to generate the average face of a bunch of faces.

Environment:

```
dlib
opencv-python
```

## 1. What's the concept of average face?

Thanks to Satya Mallick's blog: [https://learnopencv.com/average-face-opencv-c-python-tutorial/](https://learnopencv.com/average-face-opencv-c-python-tutorial/)

## 2. Requriments

```shell
cmake==3.22.0
numpy==1.21.4
scikit-image==0.19.0
dlib==19.22.1
opencv-python==4.5.4.60
```

Note: only support opencv version 4, other version need to rectify the function `cv2.estimateAffinePartial2D` in file `face_average.py`.

## 3. Usage

Please see file [tutorial.ipynb](./tutorial.ipynb) for more details.

## 4. Visulization Results

test images:

| | | |
|-------|---------------|-----|
| ![](test_image/barak-obama.jpg) | ![](test_image/bill-clinton.jpg) | ![](test_image/george-h-bush.jpg) |
| ![](test_image/george-w-bush.jpg) | ![](test_image/jimmy-carter.jpg) | ![](test_image/ronald-regan.jpg) |

average image:

![](results/example_average_face.png)

## 5. example for vggface2 

calculate the average face of dataset vggface2

first, calculate the landmark:

```shell
python vggface2landmark.py \
    --image-dir VGGFace2/train/ \
    --save-dir VGGFace2/
```

calculate each people average face

```shell
python vggface2averageface.py \
    --image-dir VGGFace2/train/ \
    --landmark-dir VGGFace2/ \
    --size 224 \ 
    --save-dir results/VGGFace2/
```

then, you will get several average face like:

| | | |
|-------|---------------|-----|
| ![](results/VGGFace2/n000138.jpg) | ![](results/VGGFace2/n000793.jpg) | ![](results/VGGFace2/n002445.jpg) |
| ![](results/VGGFace2/n000307.jpg) | ![](results/VGGFace2/n000953.jpg) | ![](results/VGGFace2/n002450.jpg) |
| ![](results/VGGFace2/n000325.jpg) | ![](results/VGGFace2/n002326.jpg) | ![](results/VGGFace2/n002649.jpg) |


## 6. Get the average face of a dataset

final, calculate the average face from the various people's average faces:

|VGGFace2-all|VGGFace2-Female|VGGFace2-Male|
|-|-|-|
| ![](results/VGGFace2-all.png) | ![](results/VGGFace2-female.png) | ![](results/VGGFace2-male.png) |

|CelebA-all|CelebA-Female|CelebA-Male|
|-|-|-|
| ![](results/CelebA-all.png) | ![](results/CelebA-female.png) | ![](results/CelebA-male.png) |


```shell
python dataset_face_average.py\
    --image-dir results/VGGFace2-Male\
    --save-dir ./results/VGGFace-male.png
```

## Acknowledge

This code is build from [Naurislv](https://github.com/Naurislv)'s project: [https://github.com/Naurislv/facial_image_averaging](https://github.com/Naurislv/facial_image_averaging)
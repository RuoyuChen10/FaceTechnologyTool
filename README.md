# Face Technology Tool

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.7.1](https://img.shields.io/badge/pytorch-1.7.1-green.svg?style=plastic)

A collection of code commonly used to process face technology.

Note that this is not the most optimal or state-of-art demo.

## Security

[![Security Status](https://www.murphysec.com/platform3/v3/badge/1611677694740168704.svg?t=1)](https://www.murphysec.com/accept?code=57f251056a8e7c8b5ce36469616a681d&type=1&from=2&t=2)

## 1. Average Face

Compute the average face of a set of face images or face datasets

| VGGFace2 | Celeb-A |
| - | - |
|![](./AverageFace/results/VGGFace2-all.png)|![](./AverageFace/results/CelebA-all.png)|

## 2. Face Recognition

Face alignment

|   |   |
|---|---|
|![](./FaceRecognition/images/jf.jpg)|![](./FaceRecognition/images/jf-detected.jpg)|

Support loss:

- softmax
- arcface
- cosface
- focal loss

we also support evidential deep learning for face recognition

![](./FaceRecognition/images/EDL.jpg)

## 3. Face Verification

AUC is often used as an evaluation index in face verification.

```
+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| 1e-06  | 1e-05  | 0.0001 | 0.001  |  0.01  |  0.1   |  0.2   |  0.4   |  0.6   |  0.8   |   1    |
+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| 0.9251 | 0.9251 | 0.9251 | 0.9331 | 0.9421 | 0.9570 | 0.9710 | 0.9780 | 0.9870 | 0.9940 | 1.0000 |
+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
ROC AUC: 0.9800934433250592
```

![](./Verification/Celeb-A2-AUC.jpg)


## 4. Grad-CAM

Visualization the model results.

| image | Grad-CAM | Part Score | Edge |
| - | - | - | - |
| ![](Grad-CAM/image/image.jpg) | ![](Grad-CAM/image/saliency.jpg) | ![](Grad-CAM/image/PartScore.jpg) | ![](Grad-CAM/image/Edge.jpg) |

## 5. Segmentation

![](./FaceSegmentation/images/mask_image.jpg)

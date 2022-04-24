# Face Technology Tool

A collection of code commonly used to process face technology.

Note that this is not the most optimal or state-of-art demo.

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
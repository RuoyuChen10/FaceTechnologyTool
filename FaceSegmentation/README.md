## Face Segmentation

We want to segment all the faces in a photo:

![](./images/Presidents.jpg)

Please download the checkpoint.

```
sh download_ckpt.sh
```

First, detect all the faces

![](./images/det_image.jpg)

Second, generate the mask

|Segementation Result | Mask |
|-|-|
|![](./images/seg_result.jpg)|![](./images/mask.jpg)|

Segmentation Result:

![](./images/mask_image.jpg)
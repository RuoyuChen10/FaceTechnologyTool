## Face Attribute Recognition

We release our pre-trained attribute net model called Face-Attributes2.pth, please download first.

[Google Drive](https://drive.google.com/drive/folders/1upsYYgIzyzRuNEGCc7uPQ1AZ6NcHiPbD)

Usage:

```python
from AttributeNet import AttributeNet

image_path = "./face.jpg"
ckpt_path = "./Face-Attributes2.pth"
desired_attribute = ["Male","Female","Young","Middle Aged","Senior","Asian","White","Black"]
device = "cuda:0"

# init model
model = AttributeNet(pretrained = ckpt_path)
model.to(device)

# set the output attribute prediction
model.set_idx_list(attribute = desired_attribute)

# pre-proccess the image
def Path_Image_Proccessing(path):
    """
    Pre-proccessing the input images
      path: single image input path
    """
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    image = cv2.imread(path)
    assert image is not None
    image = cv2.resize(image, (224,224))
    image =image.astype(np.float32) 
    image -= mean bgr 
    #H*W*C --> C*H*W 
    image =image.transpose(2,0,1) 
    image =torch.tensor(image 
    image =image.unsqueeze(0) 
    return image 
 
 input_image = Path_Image_Proccessing(image_path)
 
 # predict the attribute
 predicted = model(input_image.to(device))
```

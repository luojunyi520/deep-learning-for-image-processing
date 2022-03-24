import torch
from alexnet_model import AlexNet
from resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from model import shufflenet_v2_x1_0
# data_transform = transforms.Compose(
#     [transforms.Resize((224, 224)),
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
model = shufflenet_v2_x1_0(num_classes=4)
# model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./model-29.pth"  # "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path,map_location=torch.device('cpu')))
print(model)

# load image
img = Image.open("./per00001.jpg")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    print(im.shape)
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        # plt.imshow(im[:, :, i], cmap='gray')
        plt.imshow(im[:, :, i])
    plt.show()


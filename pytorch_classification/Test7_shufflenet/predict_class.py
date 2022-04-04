import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import shufflenet_v2_x1_0
import torchvision.models
import onnxruntime
import numpy
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image


    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = shufflenet_v2_x1_0(num_classes=4).to(device)
    # load model weights
    model_weight_path = "./model-29.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    label_cla = 0
    with torch.no_grad():
        # predict class
        # session = onnxruntime.InferenceSession('./model-29.onnx')
        # input_name = session.get_inputs()[0].name
        # output = session.run([], {input_name:img.data.numpy()})##输出为list，需要转为tensor
        # output = torch.tensor(numpy.array(output)).squeeze()
        for i in range(0,100):
            img = Image.open("../image/" + str(i) + ".jpg")
            image =img

            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            img_path = './'+str(label_cla)+str(predict_cla)+"/"
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            image.save(img_path + str(i) + ".png")

if __name__ == '__main__':
    main()

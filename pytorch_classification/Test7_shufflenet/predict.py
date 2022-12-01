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
    img_path = "./both.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

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

        output = torch.squeeze(model(img.to(device))).cpu()
        print(output.shape)

        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()




    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()

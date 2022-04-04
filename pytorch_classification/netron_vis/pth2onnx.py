import torch
# import trochvision
import torch.utils.data
import argparse
import onnxruntime
from shufflenet import shufflenet_v2_x1_0
import os
import cv2
import numpy as np
from torch.autograd import Variable
from onnxruntime.datasets import get_example
from torchvision import transforms
from PIL import Image


def main(args):
	# print("the version of torch is {}".format(torch.__version__))
	dummy_input = getInput(args.img_size)  # 获得网络的输入
	# 加载模型
	model = shufflenet_v2_x1_0(num_classes=4)
	# model = torch.load(args.model_path, map_location='cpu')
	model_weight_path = "./model/model-29.pth"  # "./resNet34.pth"
	model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
	model.eval()
	pre = model(dummy_input)
	print("the pre:{}".format(pre))
	# 保存onnx模型
	torch2onnx(args, model, dummy_input)


def getInput(img_size):
	data_transform = transforms.Compose(
		[transforms.Resize(256),
		 transforms.CenterCrop(224),
		 transforms.ToTensor(),
		 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	# load image
	img_path = "./both.png"
	assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
	img = Image.open(img_path)
	# [N, C, H, W]
	img = data_transform(img)
	# expand batch dimension
	dummy_input = torch.unsqueeze(img, dim=0)
	# input = cv2.imread(r"./both.png")
	# # input = cv2.resize(input, (img_size, img_size))  # hwc bgr
	# input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)  # hwc rgb
	# # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	# input = np.transpose(input, (2, 0, 1)).astype(np.float32)  # chw rgb
	# # input=input/255.0
	# print("befor the input[0,0,0]:{}".format(input[0, 0, 0]))
	# print("the size of input[0,...] is {}".format(input[0, ...].shape))
	# print("the size of input[1,...] is {}".format(input[1, ...].shape))
	# print("the size of input[2,...] is {}".format(input[2, ...].shape))
	# input[0, ...] = ((input[0, ...] / 255.0) - 0.485) / 0.229
	# input[1, ...] = ((input[1, ...] / 255.0) - 0.456) / 0.224
	# input[2, ...] = ((input[2, ...] / 255.0) - 0.406) / 0.225
	# print("after input[0,0,0]:{}".format(input[0, 0, 0]))
	#
	# now_image1 = Variable(torch.from_numpy(input))
	# dummy_input = now_image1.unsqueeze(0)


	return dummy_input


def torch2onnx(args, model, dummy_input):
	input_names = ['input']  # 模型输入的name
	output_names = ['output']  # 模型输出的name
	# return
	torch_out = torch.onnx._export(model, dummy_input, os.path.join("model-29.onnx"),
								   verbose=True, input_names=input_names, output_names=output_names)
	# test onnx model
	import time


	session = onnxruntime.InferenceSession('./model-29.onnx')
	# get the name of the first input of the model
	input_name = session.get_inputs()[0].name
	print('Input Name:', input_name)

	a=time.time()
	result = session.run([], {input_name: dummy_input.data.numpy()})
	# np.testing.assert_almost_equal(
	#     torch_out.data.cpu().numpy(), result[0], decimal=3)
	b=time.time()
	print(b-a)
	print("the result is {}".format(result[0]))

## [[-0.15479328 -1.970755    3.4652119  -1.0474535 ]]
###[[ 0.36135024 -2.9626298   3.7152615  -0.85364646]]
# 结果对比--有点精度上的损失
# pytorch tensor([[ 5.8738, -5.4470]], device='cuda:0')
# onnx [ 5.6525207 -5.2962923]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="PyTorch model to onnx and ncnn")
	parser.add_argument('--model_path', type=str,
						default=r"./model/model-29.pth",
						help="For training from one model_file")
	parser.add_argument('--save_model_path', type=str,
						default=r"./model",
						help="For training from one model_file")
	parser.add_argument('--onnx_model_path', type=str,
						default=r"./model",
						help="For training from one model_file")
	parser.add_argument('--img_size', type=int, default=256,
						help="the image size of model input")
	args = parser.parse_args()
	main(args)

import netron
import onnxruntime
import numpy as np
from PIL import Image
import cv2

netron.start('./net.onnx')
test_image = np.asarray(Image.open('both.png').convert('L'),dtype='float32') /255.
test_image = cv2.resize(np.array(test_image),(224,224),interpolation = cv2.INTER_CUBIC)
test_image = test_image[np.newaxis,np.newaxis,:,:]
session = onnxruntime.InferenceSession('./net.onnx')
outputs = session.run(None, {"inputs": test_image})
print(len(outputs))
print(outputs[0].shape)
#根据需要处理一下outputs[0],并可视化一下结果，看看结果是否正常

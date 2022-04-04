import onnx
from onnx import shape_inference

model = r'./model-29.onnx'
#上一步保存好的onnx格式的模型路径
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model)), model)
#增加节点的shape信息
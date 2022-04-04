

import numpy as np
from PIL import Image
import os

path='./severe/'
newpath='./severe1/'
def turnto24(path):
    files = os.listdir(path)
    files = np.sort(files)
    i=0
    for f in files:
        imgpath = path + f
        img=Image.open(imgpath)
        if len(img.split())!=3:
            img=img.convert('RGB')
            dirpath = newpath
            file_name, file_extend = os.path.splitext(f)
            dst = os.path.join(os.path.abspath(dirpath), file_name + '.jpg')
            img.save(dst)

turnto24(path)


# img=Image.open('./none/1064.jpg')
# print(len(img.split()))
#
# img=img.convert('RGB')
# # 直接就输出图像的通道数了
# print(len(img.split()))



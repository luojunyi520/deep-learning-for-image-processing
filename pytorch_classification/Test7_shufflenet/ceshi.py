import os
label_cla=0
predict_cla=0
from PIL import Image
origin_path = "./image"
count = os.listdir(origin_path+'/')
for i in range(0, len(count)):
    path = os.path.join(origin_path, count[i])
    img = Image.open(path)
    img_path = './'+str(label_cla)+'_'+str(predict_cla)+'/'
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    img.save(img_path+os.path.basename(path))
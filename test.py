import os
import json
from PIL import Image
from glob import glob
import torch
import torchvision
import subprocess
import shlex
from collections import Counter

from modules.image_util import imShow, imshow, infer_image_with_EN, transform_img


### SETUP ###
device = "cuda:0" if torch.cuda.is_available() else "cpu"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.chdir("./darknet/")
os.system("make")
os.chdir("..")

yolo_data = "data/Sock.data"
yolo_cfg = "cfg/yolov4-tiny-custom.cfg"
yolo_weights = "backup/yolov4-tiny-custom_best.weights"
yolo_result_json = "prediction.json"

os.chdir("./EfficientNet-PyTorch/")
from efficientnet_pytorch import EfficientNet
os.chdir("..")
en_weight_path = "./EfficientNet-PyTorch/logs/trial0/sock_clsfy_best.pt"
cls_classes = 10
en_model = EfficientNet.from_name('efficientnet-b0', num_classes=cls_classes)
en_model.load_state_dict(torch.load(en_weight_path))
en_model.to(device)


### Bring the Image ###
upload_dir = "upload_img/"
image_path = glob(f"./{upload_dir}*.jpg")[0]
print("Input Image Path:", image_path)
imShow(image_path) ###########
img = Image.open(image_path)
width, height = img.size


### YOLO Detection ###
os.chdir("./darknet")
#import subprocess
#import shlex
text = f"./darknet detector test {yolo_data} {yolo_cfg} {yolo_weights} {image_path} -ext_output -dont_show -out {yolo_result_json}"
process = subprocess.Popen(shlex.split(text),
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, _ = process.communicate()
print("Detected Socks By Yolo:")
imShow("predictions.jpg") ###########
with open(yolo_result_json, "r") as f:
    yolo_inference_data = json.load(f)
bboxes = yolo_inference_data[0]['objects']


### EfficientNet Classification ###
os.chdir("../EfficientNet-PyTorch")
class_names = {
    '0': 'houndstooth',
    '1': 'beigethombrowne',
    '2': 'bluestripe',
    '3': 'wishsocks',
    '4': 'commedegarcon',
    '5': 'indigogolf',
    '6': 'whitegolf',
    '7': 'blackstripe',
    '8': 'moomin',
    '9': 'tartan'
}
class_names_list = list(class_names.values())

socks, names = [], []
for obj in bboxes:
    bbox = obj["relative_coordinates"]
    xmid, ymid, w, h = bbox["center_x"], bbox["center_y"], bbox["width"], bbox["height"]
    xmin, xmax = round((xmid - w / 2) * width), round((xmid + w / 2) * width)
    ymin, ymax = round((ymid - h / 2) * height), round((ymid + h / 2) * height)
    cropped_img = img.crop((xmin, ymin, xmax, ymax))
    imm, prediction = infer_image_with_EN(en_model, cropped_img, device, class_names_list)
    imm = imm.squeeze()
    socks.append(imm)
    names.append(prediction)
out = torchvision.utils.make_grid(socks)
imshow(out, title=names) #########


### Count Socks ###
count = dict(Counter(names))
paired, unpaired, overdetected = [], [], []
for k, v in count.items():
    if v == 2:
        paired.append(k)
    elif v == 1:
        unpaired.append(k)
    else: # v>3
        overdetected.append(k)
undetected = list(set(class_names_list.copy()) - set(paired + unpaired + overdetected))

count_result = f"""
Paired Socks      : {paired}
Unpaired Socks    : {unpaired}
Overdetected Socks: {overdetected}
Undetected Socks  : {undetected}
"""
print(count_result)
if unpaired + overdetected == [] and len(paired + undetected) == cls_classes:
    print("All paired!")
else:
    print("Go and find missing socks: \n", unpaired)
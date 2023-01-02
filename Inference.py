from PIL import Image
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import torch

from utils.openpose_net import OpenPoseNet

net = OpenPoseNet()

net_weights = torch.load(
    './weights/pose_model_scratch.pth', map_location={'cuda:0': 'cpu'})
keys = list(net_weights.keys())

weights_load = {}

for i in range(len(keys)):
    weights_load[list(net.state_dict().keys())[i]
                 ] = net_weights[list(keys)[i]]

state = net.state_dict()
state.update(weights_load)
net.load_state_dict(state)

print('### pretrained weight loaded ###')

# 자세 추정을 출력할 이미지
test_image = './data/example.jpg'
oriImg = cv2.imread(test_image)

oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
plt.imshow(oriImg)
plt.show()

size = (368, 368)
img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)
img = img.astype(np.float32) / 255.

color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]

preprocessed_img = img.copy()  # RGB

for i in range(3):
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
img = torch.from_numpy(img)

x = img.unsqueeze(0)

# 히트맵과 pafs 구하기
net.eval()
predicted_outputs, _ = net(x)

pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

pafs = cv2.resize(
    pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
heatmaps = cv2.resize(
    heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

# 6: 왼쪽 팔꿈치 -> 이미지에 합성해서 표시
heat_map = heatmaps[:, :, 6]
heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
heat_map = np.asarray(heat_map.convert('RGB'))

blend_img = cv2.addWeighted(oriImg, 0.5, heat_map, 0.5, 0)
plt.imshow(blend_img)
plt.show()

# 7: 왼쪽 손목 -> 이미지에 합성해서 표시
heat_map = heatmaps[:, :, 7]
heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
heat_map = np.asarray(heat_map.convert('RGB'))

blend_img = cv2.addWeighted(oriImg, 0.5, heat_map, 0.5, 0)
plt.imshow(blend_img)
plt.show()

# 왼쪽 팔꿈치와 왼쪽 손목을 잇는 PAF의 x벡터 -> 이미지에 합성해서 표시
paf = pafs[:, :, 24]
paf = Image.fromarray(np.uint8(cm.jet(paf)*255))
paf = np.asarray(paf.convert('RGB'))

blend_img = cv2.addWeighted(oriImg, 0.5, paf, 0.5, 0)
plt.imshow(blend_img)
plt.show()

from utils.decode_pose import decode_pose
_, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)


plt.imshow(oriImg)
plt.show()

plt.imshow(result_img)
plt.show()




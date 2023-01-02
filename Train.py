import random
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


from utils.dataloader import make_datapath_list, DataTransform, COCOkeypointsDataset

train_img_list, train_mask_list, val_img_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list(
    rootpath="./data/data/")

# 데이터 너무 커서 슬라이스
val_img_list = val_img_list[0:2000]
val_mask_list = val_mask_list[0:2000]
val_meta_list = val_meta_list[0:2000]

# Dataset
train_dataset = COCOkeypointsDataset(
    val_img_list, val_mask_list, val_meta_list, phase="train", transform=DataTransform())

# val_dataset = CocokeypointsDataset(val_img_list, val_mask_list, val_meta_list, phase="val", transform=DataTransform())

# DataLoader
batch_size = 32

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

# val_dataloader = data.DataLoader(
#    val_dataset, batch_size=batch_size, shuffle=False)


# dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
dataloaders_dict = {"train": train_dataloader, "val": None} # validation 안할 거라서 none 입력

from utils.openpose_net import OpenPoseNet
net = OpenPoseNet()

class OpenPoseLoss(nn.Module):

    def __init__(self):
        super(OpenPoseLoss, self).__init__()

    def forward(self, saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask):
        """

        Parameters
        ----------
        saved_for_loss : OpenPoseNet 출력

        heatmap_target : [num_batch, 19, 46, 46] # 히트맵 정답 어노테이션
        heatmap_mask : [num_batch, 19, 46, 46] # 히트맵 마스크
        paf_target : [num_batch, 38, 46, 46] # paf 정답 어노테이션
        paf_mask : [num_batch, 38, 46, 46] # paf 마스크

        Returns
        -------
        loss : 텐서, 로스
        """

        total_loss = 0
        for j in range(6):

            # PAFs （paf_mask=0 은 무시)
            pred1 = saved_for_loss[2 * j] * paf_mask
            gt1 = paf_target.float() * paf_mask

            # heatmaps
            pred2 = saved_for_loss[2 * j + 1] * heat_mask
            gt2 = heatmap_target.float()*heat_mask

            total_loss += F.mse_loss(pred1, gt1, reduction='mean') + \
                F.mse_loss(pred2, gt2, reduction='mean')

        return total_loss


criterion = OpenPoseLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2,
                      momentum=0.9,
                      weight_decay=0.0001)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device ： ", device)

    net.to(device)

    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloaders_dict["train"].dataset)
    batch_size = dataloaders_dict["train"].batch_size

    iteration = 1

    for epoch in range(num_epochs):

        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                optimizer.zero_grad()
                print('（train）')

            else:
                continue
                # net.eval()
                # print('-------------')
                # print('（val）')

            for imges, heatmap_target, heat_mask, paf_target, paf_mask in dataloaders_dict[phase]:

                # 미니 배치가 1일 경우엔 오류 남.
                if imges.size()[0] == 1:
                    continue

                # GPU
                imges = imges.to(device)
                heatmap_target = heatmap_target.to(device)
                heat_mask = heat_mask.to(device)
                paf_target = paf_target.to(device)
                paf_mask = paf_mask.to(device)

                # optimizer
                optimizer.zero_grad()

                #（forward）
                with torch.set_grad_enabled(phase == 'train'):
                    # (out6_1, out6_2)
                    _, saved_for_loss = net(imges)

                    loss = criterion(saved_for_loss, heatmap_target,
                                     heat_mask, paf_target, paf_mask)
                    del saved_for_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10 iter 마다 1번씩 loss 출력
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('iterations {} || Loss: {:.4f} || per 10 iter: {:.4f} sec.'.format(
                                iteration, loss.item()/batch_size, duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    # else:
                        #epoch_val_loss += loss.item()

        # epoch
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_train_Loss:{:.4f} ||Epoch_val_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss/num_train_imgs, 0))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    # 마지막 nn 저장
    torch.save(net.state_dict(), './data/weights/openpose_net_' +
               str(epoch+1) + '.pth')

num_epochs = 5
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)


# Multi-Person-2D-Pose-Estimation

- 목표: 딥러닝으로 1명 또는 다수의 사람들에 대해 모션을 추론하고 히트맵 matplotlib 으로 표현하기
- 예시) 

<img width="782" alt="Screen-Shot-2019-04-11-at-5 17 56-PM" src="https://user-images.githubusercontent.com/50253860/210201553-30daf665-67e9-4e6d-931c-c95ccdef76bd.png">
[출처:https://nanonets.com/blog/human-pose-estimation-2d-guide/]

### 파일 구조

```
# 전체 파일 구조 
  ├── ./data                         # MS COCO dataset 
  |       ├── ./meta
  |       |      └── ./val_2014      # mask data
  |       ├── ./val_2014             # image data
  |       ├── ./weights              # trained models - weight file
  |       └── COCO.json              # val_data에 대한 info.json
  |      
  ├── utils                          # 데이터 전처리에 필요한 모듈
  |       ├── data_augmentation.py
  |       ├── decode_pose.py
  |       ├── dataloader.py
  |       └── openpose_net.py
  |      
  ├── Data_Loader.py                 # 데이터 전처리
  ├── Net.py                         # Utility function for calculating model size 
  |
  ├── train.py                       # 학습
  ├── inference.py                   # 추론 -> weights.pth 저장
  |
  ├── README.md                      # This file
  ├── example.jpg                    # sample img to show results
  └── requirements.txt               # External module dependencies(필요한 라이브러리)
```

```
필요한 수동 다운로드 목록

- mask.tar.gz : https://www.dropbox.com/s/bd9ty7b4fqd5ebf/mask.tar.gz?dl=0 
- coco.json : https://www.dropbox.com/s/0sj2q24hipiiq5t/COCO.json?dl=0
- pose_model_scratch.pth : https://www.dropbox.com/s/5v654d2u65fuvyr/pose_model_scratch.pth?dl=0
```


------------------
### 데이터 전처리

**Step 1. Image resize**  

이미지 크기를 줄여줌.  
<img src="https://user-images.githubusercontent.com/50253860/210203625-f42315f0-f9c9-4f24-8a10-267ed430c300.png" width="580" height="200"/>

```
+ PAFs: 부위 간에 연결성을 나타내는 지표. 
ex) 왼쪽 손목~왼쪽 팔꿈치 사이의 확률 부분을 잇는 선분
```

**Step 2. Input : Image -> Net**  

전처리된 이미지를 신경망에 입력.  
신체 부위(19개)에 해당하는 배열 출력 = (19 * 368 * 368)  
신체 부위(19개)+PAFs 벡터 좌표(19개)에 해당하는 배열 출력 = (38 * 368 * 368)  

**Step 3. Output: Coordinate**  

출력 결과에서 각 부위별 좌표, PaFs 정보를 구함.  
이미지 크기는 원래대로 되돌림.  

<img src="https://user-images.githubusercontent.com/50253860/210204080-f23fcf0b-4846-4d40-8628-f6afe37b633c.png" width="580" height="200"/>


------------------

### 데이터셋 다운로드, 데이터로더

#### Dataset : MSCOCO 2017 dataset 사용
(*COCO에서 제공하는 train dataset이 너무 큰 관계로, validation dataset만 다운받아서 train dataset으로 사용.   대신 validation loader는 사용하지 않음)
  
| |데이터|설명|
|---|------|---|
|(1)|Image data|인풋으로 들어가는 입력 이미지|
|(2)|Mask data|화면에 육안으로 보이지만, 어노테이션 데이터가 없는 인물 정보를 검게 칠하는 것.   (=> 따라서 mask image와 input image는 size 같음)|
|(3)|Annotation|pose의 정답 데이터|

--------------------
### OpenPoseNet 구현

- Feature 모듈과 스테이지 1~6, 총 7개의 모듈로 이루어진 OpenPoseNet 구성.

<details>
<summary>Model Summary</summary>
<div markdown="1">       

```
Input shape: 3 * 368 * 368 (resized)

 ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [32, 64, 368, 368]           1,792
              ReLU-2         [32, 64, 368, 368]               0
            Conv2d-3         [32, 64, 368, 368]          36,928
              ReLU-4         [32, 64, 368, 368]               0
         MaxPool2d-5         [32, 64, 184, 184]               0
            Conv2d-6        [32, 128, 184, 184]          73,856
              ReLU-7        [32, 128, 184, 184]               0
            Conv2d-8        [32, 128, 184, 184]         147,584
              ReLU-9        [32, 128, 184, 184]               0
        MaxPool2d-10          [32, 128, 92, 92]               0
           Conv2d-11          [32, 256, 92, 92]         295,168
             ReLU-12          [32, 256, 92, 92]               0
           Conv2d-13          [32, 256, 92, 92]         590,080
             ReLU-14          [32, 256, 92, 92]               0
           Conv2d-15          [32, 256, 92, 92]         590,080
             ReLU-16          [32, 256, 92, 92]               0
           Conv2d-17          [32, 256, 92, 92]         590,080
             ReLU-18          [32, 256, 92, 92]               0
        MaxPool2d-19          [32, 256, 46, 46]               0
           Conv2d-20          [32, 512, 46, 46]       1,180,160
             ReLU-21          [32, 512, 46, 46]               0
           Conv2d-22          [32, 512, 46, 46]       2,359,808
             ReLU-23          [32, 512, 46, 46]               0
           Conv2d-24          [32, 256, 46, 46]       1,179,904
             ReLU-25          [32, 256, 46, 46]               0
           Conv2d-26          [32, 128, 46, 46]         295,040
             ReLU-27          [32, 128, 46, 46]               0
 OpenPose_Feature-28          [32, 128, 46, 46]               0
           Conv2d-29          [32, 128, 46, 46]         147,584
             ReLU-30          [32, 128, 46, 46]               0
           Conv2d-31          [32, 128, 46, 46]         147,584
             ReLU-32          [32, 128, 46, 46]               0
           Conv2d-33          [32, 128, 46, 46]         147,584
             ReLU-34          [32, 128, 46, 46]               0
           Conv2d-35          [32, 512, 46, 46]          66,048
             ReLU-36          [32, 512, 46, 46]               0
           Conv2d-37           [32, 38, 46, 46]          19,494
           Conv2d-38          [32, 128, 46, 46]         147,584
             ReLU-39          [32, 128, 46, 46]               0
           Conv2d-40          [32, 128, 46, 46]         147,584
             ReLU-41          [32, 128, 46, 46]               0
           Conv2d-42          [32, 128, 46, 46]         147,584
             ReLU-43          [32, 128, 46, 46]               0
           Conv2d-44          [32, 512, 46, 46]          66,048
             ReLU-45          [32, 512, 46, 46]               0
           Conv2d-46           [32, 19, 46, 46]           9,747
           Conv2d-47          [32, 128, 46, 46]       1,160,448
             ReLU-48          [32, 128, 46, 46]               0
           Conv2d-49          [32, 128, 46, 46]         802,944
             ReLU-50          [32, 128, 46, 46]               0
           Conv2d-51          [32, 128, 46, 46]         802,944
             ReLU-52          [32, 128, 46, 46]               0
           Conv2d-53          [32, 128, 46, 46]         802,944
             ReLU-54          [32, 128, 46, 46]               0
           Conv2d-55          [32, 128, 46, 46]         802,944
             ReLU-56          [32, 128, 46, 46]               0
           Conv2d-57          [32, 128, 46, 46]          16,512
             ReLU-58          [32, 128, 46, 46]               0
           Conv2d-59           [32, 38, 46, 46]           4,902
           Conv2d-60          [32, 128, 46, 46]       1,160,448
             ReLU-61          [32, 128, 46, 46]               0
           Conv2d-62          [32, 128, 46, 46]         802,944
             ReLU-63          [32, 128, 46, 46]               0
           Conv2d-64          [32, 128, 46, 46]         802,944
             ReLU-65          [32, 128, 46, 46]               0
           Conv2d-66          [32, 128, 46, 46]         802,944
             ReLU-67          [32, 128, 46, 46]               0
           Conv2d-68          [32, 128, 46, 46]         802,944
             ReLU-69          [32, 128, 46, 46]               0
           Conv2d-70          [32, 128, 46, 46]          16,512
             ReLU-71          [32, 128, 46, 46]               0
           Conv2d-72           [32, 19, 46, 46]           2,451
           Conv2d-73          [32, 128, 46, 46]       1,160,448
             ReLU-74          [32, 128, 46, 46]               0
           Conv2d-75          [32, 128, 46, 46]         802,944
             ReLU-76          [32, 128, 46, 46]               0
           Conv2d-77          [32, 128, 46, 46]         802,944
             ReLU-78          [32, 128, 46, 46]               0
           Conv2d-79          [32, 128, 46, 46]         802,944
             ReLU-80          [32, 128, 46, 46]               0
           Conv2d-81          [32, 128, 46, 46]         802,944
             ReLU-82          [32, 128, 46, 46]               0
           Conv2d-83          [32, 128, 46, 46]          16,512
             ReLU-84          [32, 128, 46, 46]               0
           Conv2d-85           [32, 38, 46, 46]           4,902
           Conv2d-86          [32, 128, 46, 46]       1,160,448
             ReLU-87          [32, 128, 46, 46]               0
           Conv2d-88          [32, 128, 46, 46]         802,944
             ReLU-89          [32, 128, 46, 46]               0
           Conv2d-90          [32, 128, 46, 46]         802,944
             ReLU-91          [32, 128, 46, 46]               0
           Conv2d-92          [32, 128, 46, 46]         802,944
             ReLU-93          [32, 128, 46, 46]               0
           Conv2d-94          [32, 128, 46, 46]         802,944
             ReLU-95          [32, 128, 46, 46]               0
           Conv2d-96          [32, 128, 46, 46]          16,512
             ReLU-97          [32, 128, 46, 46]               0
           Conv2d-98           [32, 19, 46, 46]           2,451
           Conv2d-99          [32, 128, 46, 46]       1,160,448
            ReLU-100          [32, 128, 46, 46]               0
          Conv2d-101          [32, 128, 46, 46]         802,944
            ReLU-102          [32, 128, 46, 46]               0
          Conv2d-103          [32, 128, 46, 46]         802,944
            ReLU-104          [32, 128, 46, 46]               0
          Conv2d-105          [32, 128, 46, 46]         802,944
            ReLU-106          [32, 128, 46, 46]               0
          Conv2d-107          [32, 128, 46, 46]         802,944
            ReLU-108          [32, 128, 46, 46]               0
          Conv2d-109          [32, 128, 46, 46]          16,512
            ReLU-110          [32, 128, 46, 46]               0
          Conv2d-111           [32, 38, 46, 46]           4,902
          Conv2d-112          [32, 128, 46, 46]       1,160,448
            ReLU-113          [32, 128, 46, 46]               0
          Conv2d-114          [32, 128, 46, 46]         802,944
            ReLU-115          [32, 128, 46, 46]               0
          Conv2d-116          [32, 128, 46, 46]         802,944
            ReLU-117          [32, 128, 46, 46]               0
          Conv2d-118          [32, 128, 46, 46]         802,944
            ReLU-119          [32, 128, 46, 46]               0
          Conv2d-120          [32, 128, 46, 46]         802,944
            ReLU-121          [32, 128, 46, 46]               0
          Conv2d-122          [32, 128, 46, 46]          16,512
            ReLU-123          [32, 128, 46, 46]               0
          Conv2d-124           [32, 19, 46, 46]           2,451
          Conv2d-125          [32, 128, 46, 46]       1,160,448
            ReLU-126          [32, 128, 46, 46]               0
          Conv2d-127          [32, 128, 46, 46]         802,944
            ReLU-128          [32, 128, 46, 46]               0
          Conv2d-129          [32, 128, 46, 46]         802,944
            ReLU-130          [32, 128, 46, 46]               0
          Conv2d-131          [32, 128, 46, 46]         802,944
            ReLU-132          [32, 128, 46, 46]               0
          Conv2d-133          [32, 128, 46, 46]         802,944
            ReLU-134          [32, 128, 46, 46]               0
          Conv2d-135          [32, 128, 46, 46]          16,512
            ReLU-136          [32, 128, 46, 46]               0
          Conv2d-137           [32, 38, 46, 46]           4,902
          Conv2d-138          [32, 128, 46, 46]       1,160,448
            ReLU-139          [32, 128, 46, 46]               0
          Conv2d-140          [32, 128, 46, 46]         802,944
            ReLU-141          [32, 128, 46, 46]               0
          Conv2d-142          [32, 128, 46, 46]         802,944
            ReLU-143          [32, 128, 46, 46]               0
          Conv2d-144          [32, 128, 46, 46]         802,944
            ReLU-145          [32, 128, 46, 46]               0
          Conv2d-146          [32, 128, 46, 46]         802,944
            ReLU-147          [32, 128, 46, 46]               0
          Conv2d-148          [32, 128, 46, 46]          16,512
            ReLU-149          [32, 128, 46, 46]               0
          Conv2d-150           [32, 19, 46, 46]           2,451
          Conv2d-151          [32, 128, 46, 46]       1,160,448
            ReLU-152          [32, 128, 46, 46]               0
          Conv2d-153          [32, 128, 46, 46]         802,944
            ReLU-154          [32, 128, 46, 46]               0
          Conv2d-155          [32, 128, 46, 46]         802,944
            ReLU-156          [32, 128, 46, 46]               0
          Conv2d-157          [32, 128, 46, 46]         802,944
            ReLU-158          [32, 128, 46, 46]               0
          Conv2d-159          [32, 128, 46, 46]         802,944
            ReLU-160          [32, 128, 46, 46]               0
          Conv2d-161          [32, 128, 46, 46]          16,512
            ReLU-162          [32, 128, 46, 46]               0
          Conv2d-163           [32, 38, 46, 46]           4,902
          Conv2d-164          [32, 128, 46, 46]       1,160,448
            ReLU-165          [32, 128, 46, 46]               0
          Conv2d-166          [32, 128, 46, 46]         802,944
            ReLU-167          [32, 128, 46, 46]               0
          Conv2d-168          [32, 128, 46, 46]         802,944
            ReLU-169          [32, 128, 46, 46]               0
          Conv2d-170          [32, 128, 46, 46]         802,944
            ReLU-171          [32, 128, 46, 46]               0
          Conv2d-172          [32, 128, 46, 46]         802,944
            ReLU-173          [32, 128, 46, 46]               0
          Conv2d-174          [32, 128, 46, 46]          16,512
            ReLU-175          [32, 128, 46, 46]               0
          Conv2d-176           [32, 19, 46, 46]           2,451
================================================================
Total params: 52,311,446
Trainable params: 52,311,446
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 49.59
Forward/backward pass size (MB): 3430.20
Params size (MB): 199.55
Estimated Total Size (MB): 3679.34
----------------------------------------------------------------
```
</div>
</details>

------------------
### Feature 모듈 구현

VGG-19의 최초 열 번째 합성곱 층까지 구조를 그대로 사용
+ 새로운 합성곱 층 2개 준비


### Stage 모듈 구성  
  
스테이지 1)  
 블록 1_1은 PAFs 측에 대응  
 블록 1_2는 히트맵 측에 대응  
   
스테이지 n=2~6)  
 블록 n_1은 PAFs 측에 대응  
 블록 n_2는 히트맵 측에 대응  


------------------
### 학습(Training)

```
# options
epochs = 10
batch size = 16
optimizer : Adam
```
- train_loader는 validation dataset으로 구성한다.(파일 크기 때문)
= validation_loader = None.

```
# 실행 화면
Device ：  cuda:0
-------------
Epoch 1/5
-------------
（train）
iterations 10 || Loss: 0.0091 || per 10 iter: 42.5833 sec.
iterations 20 || Loss: 0.0078 || per 10 iter: 54.0288 sec.
iterations 30 || Loss: 0.0069 || per 10 iter: 32.3135 sec.
iterations 40 || Loss: 0.0059 || per 10 iter: 20.9394 sec.
iterations 50 || Loss: 0.0047 || per 10 iter: 20.6751 sec.
iterations 60 || Loss: 0.0045 || per 10 iter: 21.4848 sec.
-------------
epoch 1 || Epoch_train_Loss:0.0069 ||Epoch_val_Loss:0.0000
timer:  208.4242 sec.
-------------
Epoch 2/5
-------------
（train）
iterations 70 || Loss: 0.0036 || per 10 iter: 42.9272 sec.
iterations 80 || Loss: 0.0033 || per 10 iter: 20.2265 sec.
iterations 90 || Loss: 0.0026 || per 10 iter: 20.6426 sec.
iterations 100 || Loss: 0.0026 || per 10 iter: 21.1457 sec.
iterations 110 || Loss: 0.0022 || per 10 iter: 20.3675 sec.
iterations 120 || Loss: 0.0020 || per 10 iter: 20.9297 sec.
-------------
epoch 2 || Epoch_train_Loss:0.0028 ||Epoch_val_Loss:0.0000
timer:  162.8061 sec.
.
.
.

```

### loss function

각 히트맵과 PAF에서 각 픽셀 값이 정답 데이터값과 얼마나 가까운 값이 되는지 픽셍별 값을 regression 한다.  
평균 제곱 오차 함수 ```F.mse_loss()``` 사용.  
네트워크 모델 전체의 오차는 6개의 스테이지의 히트맵과 PAFs의 모든 오차를 더한다. 
학습이 끝난 모델은 .pth 가중치 파일로 저장.


### Inference

학습된 모델을 읽어(.pth) 화면 속 인물의 자세를 추론.  

- 원본 이미지)  
![Figure_1](https://user-images.githubusercontent.com/50253860/210199711-8eaffac1-b31f-4a36-b456-aef86f7763ed.png)

- pre-trained.pth 적용 후 이미지)  
![Figure_2](https://user-images.githubusercontent.com/50253860/210199715-bea6321d-eb7b-4ee3-b7fc-8bf01dad30dd.png)


-------------------

- 진행 중 아쉬웠던 점:  

   + 가지고 있는 GPU 사양으로는(3050) train set 데이터가 너무 많아서 거의 1/10 정도로 slice해서 진행하였다.   
   + 추론은 (1)pre-trained.pth 파일과 (2)학습을 통해서 얻은 custom-trained.pth 파일 2가지에 대해 결과를 냈다.  
   + 가능한 epochs과 batch 사이즈에 한계가 있어서 그런지 pre-trained.pth의 성능이 더 좋았다는 점이 아쉬웠다.  
      + (메모리 초과 시 !CUDA memory Error 발생)  


- References:  

  + [Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields](https://arxiv.org/abs/1611.08050)  
  + [pytorch_Realtime_Multi-Person_Pose_Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)  
  + [YutaroOgawa/pytorch_advanced](https://github.com/YutaroOgawa/pytorch_advanced)  




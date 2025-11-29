# Dog Breed Detection

Project này về phát hiện, nhận dạng giống chó sử dụng Resnet50 và CNN từ đầu.

## 1. Chuẩn bị & clone mã nguồn từ Github

```bash
git clone https://github.com/vukhai248/dog-breed-detection.git
cd dog-breed-detection
```

---
## 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

---

## 3. Chạy file test

Trong môi trường ảo, chạy lệnh:

```bash
python main.py
```

---

## 4. Chạy ứng dụng

```bash
streamlit run app.py
```

---

## 5. Giới thiệu về dataset

Dataset được sử dụng trong project này là phân loại giống chó của đại học Stanford:
- Gồm 20k+ ảnh và 120 class gồm các giống chó khác nhau
- Cấu trúc dataset được chia làm 2 phần:

```bash
stanford-dogs-dataset
├── images
│   ├── 120 folders nhãn, mỗi folder chứa các ảnh của giống chó liên quan
├── annotations
│   ├── 120 folders nhãn, mỗi folder chứa các file annotation của giống chó liên quan
```

- Đây là link của dataset: [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data)

## 6. Model sử dụng

Hiện tại project đang được sử dụng 2 model:
- Resnet50 transfer learning
- CNN theo kiến trúc sau:

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CNN                                      [32, 120]                 --
├─Conv2d: 1-1                            [32, 32, 224, 224]        896
├─BatchNorm2d: 1-2                       [32, 32, 224, 224]        64
├─ReLU: 1-3                              [32, 32, 224, 224]        --
├─Conv2d: 1-4                            [32, 32, 224, 224]        9,248
├─BatchNorm2d: 1-5                       [32, 32, 224, 224]        64
├─ReLU: 1-6                              [32, 32, 224, 224]        --
├─MaxPool2d: 1-7                         [32, 32, 112, 112]        --
├─Conv2d: 1-8                            [32, 64, 112, 112]        18,496
├─BatchNorm2d: 1-9                       [32, 64, 112, 112]        128
├─ReLU: 1-10                             [32, 64, 112, 112]        --
├─Conv2d: 1-11                           [32, 64, 112, 112]        36,928
├─BatchNorm2d: 1-12                      [32, 64, 112, 112]        128
├─ReLU: 1-13                             [32, 64, 112, 112]        --
├─MaxPool2d: 1-14                        [32, 64, 56, 56]          --
├─Conv2d: 1-15                           [32, 128, 56, 56]         73,856
├─BatchNorm2d: 1-16                      [32, 128, 56, 56]         256
├─ReLU: 1-17                             [32, 128, 56, 56]         --
├─Conv2d: 1-18                           [32, 128, 56, 56]         147,584
├─BatchNorm2d: 1-19                      [32, 128, 56, 56]         256
├─ReLU: 1-20                             [32, 128, 56, 56]         --
├─MaxPool2d: 1-21                        [32, 128, 28, 28]         --
├─Conv2d: 1-22                           [32, 256, 28, 28]         295,168
├─BatchNorm2d: 1-23                      [32, 256, 28, 28]         512
├─ReLU: 1-24                             [32, 256, 28, 28]         --
├─Conv2d: 1-25                           [32, 256, 28, 28]         590,080
├─BatchNorm2d: 1-26                      [32, 256, 28, 28]         512
├─ReLU: 1-27                             [32, 256, 28, 28]         --
├─MaxPool2d: 1-28                        [32, 256, 14, 14]         --
├─Conv2d: 1-29                           [32, 512, 14, 14]         1,180,160
├─BatchNorm2d: 1-30                      [32, 512, 14, 14]         1,024
├─ReLU: 1-31                             [32, 512, 14, 14]         --
├─Conv2d: 1-32                           [32, 512, 14, 14]         2,359,808
├─BatchNorm2d: 1-33                      [32, 512, 14, 14]         1,024
├─ReLU: 1-34                             [32, 512, 14, 14]         --
├─MaxPool2d: 1-35                        [32, 512, 7, 7]           --
├─AdaptiveAvgPool2d: 1-36                [32, 512, 1, 1]           --
├─Flatten: 1-37                          [32, 512]                 --
├─Linear: 1-38                           [32, 1024]                525,312
├─BatchNorm1d: 1-39                      [32, 1024]                2,048
├─ReLU: 1-40                             [32, 1024]                --
├─Dropout: 1-41                          [32, 1024]                --
├─Linear: 1-42                           [32, 512]                 524,800
├─BatchNorm1d: 1-43                      [32, 512]                 1,024
├─ReLU: 1-44                             [32, 512]                 --
├─Dropout: 1-45                          [32, 512]                 --
├─Linear: 1-46                           [32, 120]                 61,560
==========================================================================================
Total params: 5,830,936
Trainable params: 5,830,936
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 105.20
==========================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 3186.39
Params size (MB): 23.32
Estimated Total Size (MB): 3228.98
==========================================================================================
```
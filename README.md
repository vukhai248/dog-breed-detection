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


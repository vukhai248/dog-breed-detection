import os
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from ultralytics import YOLO
from src.models import Resnet50, CNN

# Load models
yolov5 = YOLO('yolov5s.pt')
yolov5.eval()

model_path = 'models/resnet50_finetune.pth'
resnet_model = Resnet50()  # Đổi tên biến để tránh trùng với class
checkpoint = torch.load(model_path, map_location='cpu')
resnet_model.load_state_dict(checkpoint['model_state_dict'])
resnet_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model = resnet_model.to(device)

def detect_and_classify_dog(image_path, yolo_model, pretrain_model, device, checkpoint):
    """
    Phát hiện và phân loại giống chó trong ảnh
    """
    # Đọc ảnh
    img_pil = Image.open(image_path).convert('RGB')
    img_cv2 = cv2.imread(image_path)
    
    # 1. NHẬN DIỆN (Dùng YOLO)
    results = yolo_model(img_pil)
    
    # Lấy detections từ results (phiên bản mới của ultralytics)
    # results[0].boxes chứa thông tin về các bounding boxes
    boxes = results[0].boxes
    
    # Lọc chỉ lấy lớp 'dog' (class_id = 16 trong COCO dataset)
    dog_detections = []
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        if class_id == 16:  # Dog class
            dog_detections.append({
                'bbox': box.xyxy[0].cpu().numpy(),  # [xmin, ymin, xmax, ymax]
                'confidence': float(box.conf[0])
            })
    
    if len(dog_detections) == 0:
        return "No Dog Detected", None, img_cv2

    # Lấy detection có confidence cao nhất
    best_detection = max(dog_detections, key=lambda x: x['confidence'])
    best_bbox = best_detection['bbox'].astype(int).tolist()
    confidence = best_detection['confidence']
    
    xmin, ymin, xmax, ymax = best_bbox
    
    # 2. TIỀN XỬ LÝ (Crop và Resize)
    # Cắt ảnh gốc theo Bounding Box
    cropped_img_pil = img_pil.crop((xmin, ymin, xmax, ymax))
    
    # Transform
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(cropped_img_pil).unsqueeze(0).to(device)
    
    # 3. PHÂN LOẠI (Dùng CNN)
    with torch.no_grad():
        output = pretrain_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Lấy top prediction
    top_p, top_class_idx = probabilities.topk(1, dim=0)
    
    # Lấy tên giống chó (key là integer, không phải string)
    class_idx = top_class_idx.item()
    predicted_breed = checkpoint['idx2label'][class_idx]
    breed_confidence = top_p.item()
    
    # 4. VẼ BOUNDING BOX VÀ LABEL LÊN ẢNH
    # Vẽ bounding box (màu xanh lá)
    cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    
    # Tạo label text
    label = f"{predicted_breed}: {breed_confidence:.2f}"
    
    # Tính toán kích thước text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Vẽ background cho text (màu đen với độ trong suốt)
    cv2.rectangle(img_cv2, 
                  (xmin, ymin - text_height - baseline - 10), 
                  (xmin + text_width, ymin), 
                  (0, 255, 0), 
                  -1)
    
    # Vẽ text (màu trắng)
    cv2.putText(img_cv2, 
                label, 
                (xmin, ymin - baseline - 5), 
                font, 
                font_scale, 
                (255, 255, 255), 
                thickness)
    
    return predicted_breed, best_bbox, img_cv2


def display_result(img_cv2, breed, bbox, save_path=None):
    """
    Hiển thị kết quả bằng cv2
    """
    # Hiển thị ảnh
    window_name = 'Dog Breed Detection'
    cv2.imshow(window_name, img_cv2)
    
    print(f"\n{'='*50}")
    print(f"Detected BBox: {bbox}")
    print(f"Predicted Breed: {breed}")
    print(f"{'='*50}")
    print("\nPress any key to close the window...")
    
    # Lưu ảnh nếu cần
    if save_path:
        cv2.imwrite(save_path, img_cv2)
        print(f"Result saved to: {save_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- VÍ DỤ SỬ DỤNG ---
if __name__ == "__main__":
    image_test_path = r'C:\Users\VICTUS\Desktop\create\xla_re\data\raw\images\Images\n02085620-Chihuahua\n02085620_3045.jpg'
    
    # Kiểm tra file tồn tại
    if not os.path.exists(image_test_path):
        print(f"Error: Image not found at {image_test_path}")
    else:
        try:
            # Phát hiện và phân loại
            predicted_breed, bbox, result_img = detect_and_classify_dog(
                image_test_path, 
                yolov5, 
                resnet_model, 
                device,
                checkpoint
            )
            
            # Hiển thị kết quả
            if bbox is not None:
                display_result(
                    result_img, 
                    predicted_breed, 
                    bbox,
                    save_path='result_output.jpg'  # Tùy chọn lưu ảnh
                )
            else:
                print(predicted_breed)  # "No Dog Detected"
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
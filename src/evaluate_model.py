import matplotlib.pyplot as plt
import numpy as np
import torch
import random

class ModelEvaluator:
    def __init__(self, model, dataloader, class_names, device):
        """
        Khởi tạo ModelEvaluator
        
        Args:
            model: Model đã train
            dataloader: DataLoader (val hoặc test)
            class_names: List tên các class
            device: Device (cuda/cpu)
        """
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.device = device

    def evaluate_grid(self, num_images=64):
        """
        Hiển thị grid 8x8 ảnh với nhãn thật và nhãn dự đoán
        
        Args:
            num_images: Số lượng ảnh hiển thị (mặc định 64 = 8x8)
        """
        self.model.eval()
        
        # Lấy ngẫu nhiên các ảnh từ dataloader
        all_images = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Lưu lại
                all_images.extend(inputs.cpu())
                all_labels.extend(labels.cpu())
                all_preds.extend(preds.cpu())
                
                # Dừng khi đủ ảnh
                if len(all_images) >= num_images:
                    break
        
        # Chọn ngẫu nhiên num_images ảnh
        indices = random.sample(range(len(all_images)), min(num_images, len(all_images)))
        
        # Tạo figure 8x8
        rows, cols = 8, 8
        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        fig.suptitle('Model Evaluation: Predicted vs Actual Labels', fontsize=20, y=0.995)
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(indices):
                img_idx = indices[idx]
                img = all_images[img_idx]
                true_label = all_labels[img_idx].item()
                pred_label = all_preds[img_idx].item()
                
                # Denormalize ảnh
                img = img.numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Hiển thị ảnh
                ax.imshow(img)
                ax.axis('off')
                
                # Tạo title với màu sắc
                is_correct = (true_label == pred_label)
                color = 'green' if is_correct else 'red'
                
                # Title: Pred / True
                title = f"Pred: {self.class_names[pred_label][:15]}\nTrue: {self.class_names[true_label][:15]}"
                ax.set_title(title, fontsize=8, color=color, weight='bold')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Tính accuracy
        correct = sum([1 for i in indices if all_preds[i] == all_labels[i]])
        accuracy = correct / len(indices) * 100
        print(f"\n{'='*60}")
        print(f"Accuracy trên {len(indices)} ảnh: {accuracy:.2f}%")
        print(f"Số ảnh dự đoán đúng: {correct}/{len(indices)}")
        print(f"{'='*60}")
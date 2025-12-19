import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Import hàm từ file utils của chúng ta
from utils.gradcam import GradCAM, show_cam_on_image

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Lưu ý: Trên Windows đường dẫn dùng dấu gạch chéo ngược (\) hoặc gạch chéo xuôi (/) đều được trong Python
MODEL_PATH = 'models/final_pneumonia_model.pth'
IMAGE_PATH = 'data_samples/PNEUMONIA/PNEUMONIA_test_0000.jpeg'  # Ảnh viêm phổi mẫu
# Thay đổi thành 'data_samples/NORMAL/NORMAL_test_0000.jpeg' để test ảnh bình thường

# 1. LOAD MODEL
def load_model():
    print(f"Đang tải mô hình từ: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy file model. Hãy copy file .pth vào thư mục 'models'.")
        return None

    # Khởi tạo kiến trúc ResNet50
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    
    # Load trọng số (map_location='cpu' để chạy trên máy không có GPU rời)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("✅ Load model thành công!")
        return model
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return None

# 2. XỬ LÝ ẢNH
def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return None, None

    # Đọc ảnh gốc bằng OpenCV để hiển thị đẹp
    img_cv2 = cv2.imread(image_path)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_cv2 = cv2.resize(img_cv2, (224, 224))
    
    # Chuẩn bị Tensor cho model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img_cv2).unsqueeze(0)
    return input_tensor, img_cv2

# 3. CHẠY TEST
def main():
    # A. Load Model
    model = load_model()
    if model is None: return

    # B. Đọc ảnh
    input_tensor, original_img = process_image(IMAGE_PATH)
    if input_tensor is None: return

    # C. Dự đoán
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    class_names = ['BÌNH THƯỜNG', 'VIÊM PHỔI']
    result = class_names[pred.item()]
    confidence = conf.item() * 100
    
    print("-" * 30)
    print(f"ẢNH: {IMAGE_PATH}")
    print(f"KẾT QUẢ: {result}")
    print(f"ĐỘ TIN CẬY: {confidence:.2f}%")
    print("-" * 30)

    # D. Chạy Grad-CAM (Lấy từ file utils)
    grad_cam = GradCAM(model)
    mask, _, _ = grad_cam(input_tensor)
    
    # Hiển thị đẹp (Lấy từ file utils)
    # Chuẩn hóa ảnh gốc về [0,1] cho hàm hiển thị
    img_float = np.float32(original_img) / 255
    visualization = show_cam_on_image(img_float, mask, alpha=0.5)

    # E. Vẽ hình
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Ảnh Gốc")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"AI Chẩn đoán: {result}\n({confidence:.1f}%)")
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()
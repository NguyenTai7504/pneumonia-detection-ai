import hashlib
import numpy as np
from PIL import Image

class ImageValidator:
    """Kiểm tra tính hợp lệ và trùng lặp của ảnh"""
    
    def __init__(self):
        self.uploaded_hashes = set()  # Lưu hash các ảnh đã upload
        
    def check_duplicate(self, image):
        """
        Kiểm tra ảnh có trùng lặp không (sử dụng MD5 hash)
        
        Args:
            image: PIL Image
        
        Returns:
            bool: True nếu trùng, False nếu không trùng
        """
        # Hash chính xác - byte-by-byte
        img_bytes = image.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        
        if img_hash in self.uploaded_hashes:
            return True
        else:
            self.uploaded_hashes.add(img_hash)
            return False
    
    def reset(self):
        """Xóa lịch sử ảnh đã upload"""
        self.uploaded_hashes.clear()


def load_xray_detector_model(model_path='models/xray_detector_resnet18_v2_BEST.pth'):
    """
    Load model phát hiện ảnh X-ray phổi (ResNet18)
    Model có 2 class: [0] = X-ray phổi, [1] = Không phải X-ray phổi
    
    Args:
        model_path: Đường dẫn file model
    
    Returns:
        model: PyTorch model đã load weights hoặc None nếu lỗi
    """
    import torch
    import torch.nn as nn
    from torchvision import models
    import os
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model: {model_path}")
        return None
    
    # Khởi tạo ResNet18
    model = models.resnet18(weights=None)
    
    # Load model để xác định kiến trúc
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Kiểm tra xem có phải fc layer đơn giản không
        if 'fc.weight' in state_dict and 'fc.0.weight' not in state_dict:
            # Classifier đơn giản: chỉ có 1 Linear layer
            fc_weight_shape = state_dict['fc.weight'].shape
            num_classes = fc_weight_shape[0]
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            print(f"✅ Model xray detector: Simple Linear classifier ({num_classes} classes)")
        
        elif 'fc.0.weight' in state_dict:
            # Classifier phức tạp: có Sequential
            fc_weight_shape = state_dict['fc.0.weight'].shape
            hidden_size = fc_weight_shape[0]
            fc_last_weight_shape = state_dict['fc.3.weight'].shape
            num_classes = fc_last_weight_shape[0]
            num_ftrs = model.fc.in_features
            
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, num_classes)
            )
            print(f"✅ Model xray detector: Sequential classifier ({num_classes} classes)")
        
        else:
            print("❌ Không nhận diện được kiến trúc model")
            return None
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Lỗi khi load model xray detector: {e}")
        return None


def is_xray_image(model, image, threshold=0.5):
    """
    Kiểm tra ảnh có phải X-ray phổi không
    
    Args:
        model: Model xray detector đã load
        image: PIL Image
        threshold: Ngưỡng confidence (0-1), mặc định 0.5
    
    Returns:
        tuple: (is_xray: bool, confidence: float)
    """
    import torch
    from torchvision import transforms
    
    if model is None:
        return True, 1.0  # Nếu không có model, bỏ qua kiểm tra
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB nếu cần
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        
        # Nếu model chỉ có 1 output (binary classification với sigmoid)
        if output.shape[1] == 1:
            xray_prob = torch.sigmoid(output[0][0]).item()
        else:
            # Multi-class với softmax
            import torch.nn.functional as F
            probs = F.softmax(output, dim=1)
            xray_prob = probs[0][0].item()  # Class 0 = X-ray phổi
    
    is_xray = xray_prob >= threshold
    return is_xray, xray_prob

import torch
import torch.nn.functional as F
import numpy as np
import cv2

# --- CLASS GRAD-CAM (Logic lõi) ---
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_blobs = []
        self.gradient = None
        # Hook vào layer4 (lớp cuối ResNet50)
        self.model.layer4.register_forward_hook(self.save_feature)
        self.model.layer4.register_full_backward_hook(self.save_gradient)

    def save_feature(self, module, input, output):
        self.feature_blobs.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        self.feature_blobs = []
        self.gradient = None
        
        output = self.model(x)
        probs = F.softmax(output, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        score = output[:, class_idx]
        score.backward()
        
        gradients = self.gradient.data.cpu().numpy()[0]
        activations = self.feature_blobs[0].data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        
        return cam, class_idx, probs[0, class_idx].item()

# --- HÀM HIỂN THỊ ĐẸP (Spotlight / Anti-glare) ---
def show_cam_on_image(img: np.ndarray, mask: np.ndarray, alpha=0.5):
    # 1. Tạo Mask cơ thể (Otsu)
    gray_img = cv2.cvtColor(np.uint8(255 * img), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) 
    enhanced_gray = clahe.apply(gray_img)
    _, body_mask = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_body_mask = np.zeros_like(body_mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_body_mask, [c], -1, 255, thickness=cv2.FILLED)
    else:
        clean_body_mask = body_mask

    clean_body_mask = clean_body_mask.astype(float) / 255.0
    clean_body_mask = cv2.GaussianBlur(clean_body_mask, (15, 15), 0)

    # 2. Xử lý ảnh nền (CLAHE RGB)
    enhanced_img_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    enhanced_img_rgb = enhanced_img_rgb.astype(float) / 255.0

    # 3. Xử lý Heatmap
    heatmap_clean = np.where(mask > 0.2, mask, 0)
    heatmap_final = heatmap_clean * clean_body_mask

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_final), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255
    heatmap_colored = heatmap_colored[..., ::-1] 
    
    # 4. Trộn màu (Công thức chống chói)
    cam = np.copy(enhanced_img_rgb)
    heatmap_indices = heatmap_final > 0
    
    cam[heatmap_indices] = (cam[heatmap_indices] * (1 - alpha)) + (heatmap_colored[heatmap_indices] * 0.8 * alpha)
    cam[~heatmap_indices] = cam[~heatmap_indices] * 0.85 # Làm tối nền
    
    return np.uint8(255 * np.clip(cam, 0, 1))
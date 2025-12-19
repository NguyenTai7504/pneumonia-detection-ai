import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# Import Grad-CAM từ utils (đã tối ưu)
from utils.gradcam import GradCAM, show_cam_on_image

# --- 2. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(
    page_title="Hỗ Trợ Chẩn Đoán Viêm Phổi - AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Medical Professional Theme
st.markdown("""
<style>
    /* Clean medical interface */
    .main {
        background-color: #f8f9fa;
    }
    
    .header-box {
        background-color: white;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        border-bottom: 3px solid #0066cc;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .result-positive {
        background-color: #fff5f5;
        border: 2px solid #e53e3e;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .result-negative {
        background-color: #f0fdf4;
        border: 2px solid #10b981;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .clinical-note {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .info-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #0052a3;
        box-shadow: 0 2px 8px rgba(0,102,204,0.3);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* File uploader - Complete overlay approach */
    [data-testid="stFileUploader"] {
        position: relative;
    }
    [data-testid="stFileUploader"] label {
        display: none !important;
    }
    /* Hide original uploader completely */
    [data-testid="stFileUploader"] section {
        position: relative;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        min-height: 120px;
    }
    [data-testid="stFileUploader"] section > div,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section button {
        opacity: 0;
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        cursor: pointer;
        z-index: 2;
    }
    /* Custom overlay UI */
    [data-testid="stFileUploader"] section::before {
        content: "";
        display: block;
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border: 2px dashed #94a3b8;
        border-radius: 8px;
        background-color: #f8fafc;
        transition: all 0.2s;
        z-index: 1;
        pointer-events: none;
    }
    [data-testid="stFileUploader"] section:hover::before {
        border-color: #0066cc;
        background-color: #f1f5f9;
    }
    /* Custom text and icon */
    [data-testid="stFileUploader"] section::after {
        content: "Kéo thả ảnh X-quang hoặc nhấn để chọn\\AJPG, PNG, JPEG • Tối đa 200MB";
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        transform: translateY(-50%);
        white-space: pre-line;
        line-height: 1.8;
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 500;
        z-index: 1;
        pointer-events: none;
        padding: 0 1rem;
    }
    /* SVG Upload Icon */
    [data-testid="stFileUploader"]::before {
        content: "";
        position: absolute;
        top: 25px;
        left: 50%;
        transform: translateX(-50%);
        width: 48px;
        height: 48px;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='%230066cc' viewBox='0 0 24 24'%3E%3Cpath d='M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z'/%3E%3Cpath d='M21 19.5c0 .83-.67 1.5-1.5 1.5h-15C3.67 21 3 20.33 3 19.5v-15C3 3.67 3.67 3 4.5 3h15c.83 0 1.5.67 1.5 1.5v15z' fill='none' stroke='%230066cc' stroke-width='1.5'/%3E%3C/svg%3E");
        background-size: contain;
        background-repeat: no-repeat;
        z-index: 1;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Khởi tạo kiến trúc ResNet50
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    
    # Tìm file model (Ưu tiên thư mục models/)
    model_paths = [
        'models/final_pneumonia_model.pth',  # Đường dẫn chuẩn
        'final_pneumonia_model.pth',  # Thư mục gốc
        '/content/drive/MyDrive/Pneumonia_ResNet50_Project/FineTuning_Phase/best_finetuned_checkpoint.pth'  # Colab
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        st.error(f"❌ Không tìm thấy file model")
        st.info("Hãy đặt file .pth vào thư mục `models/` hoặc cùng thư mục với file app.py")
        return None

    # Load trọng số (map_location='cpu' để chạy mọi nơi)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Lỗi khi đọc file model: {e}")
        return None

model = load_model()

# --- 4. HÀM XỬ LÝ ẢNH ---
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 5. GRAD-CAM NÂNG CAO (Anti-Glare + Otsu Mask) ---
def run_gradcam_advanced(model, image, alpha=0.5):
    """
    Chạy Grad-CAM với code tối ưu nhất (Anti-Glare, Otsu Masking)
    Giữ nguyên kích thước ảnh gốc để tránh kéo dãn
    """
    # Lưu kích thước gốc
    original_size = image.size  # (width, height)
    
    # Resize để inference (224x224)
    rgb_img_224 = np.array(image.resize((224, 224)))
    rgb_img_float_224 = np.float32(rgb_img_224) / 255
    
    # Preprocess cho model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(rgb_img_224).unsqueeze(0)
    
    # Chạy Grad-CAM (sử dụng class đã tối ưu từ utils)
    grad_cam = GradCAM(model)
    mask, class_idx, prob = grad_cam(input_tensor)
    
    # Resize mask về kích thước gốc
    mask_resized = cv2.resize(mask, original_size)
    
    # Ảnh gốc với kích thước gốc
    rgb_img_original = np.array(image)
    rgb_img_float_original = np.float32(rgb_img_original) / 255
    
    # Hiển thị với hàm Anti-Glare trên ảnh kích thước gốc
    visualization = show_cam_on_image(rgb_img_float_original, mask_resized, alpha=alpha)
    
    return visualization

# --- 6. GIAO DIỆN CHÍNH ---
st.markdown("""
<div class='header-box' style='text-align: center;'>
    <h2 style='color: #0066cc; margin: 0; font-weight: 600;'>🩺 Hệ Thống Phát Hiện Viêm Phổi</h2>
    <p style='color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.95rem;'>
        AI Hỗ Trợ Chẩn Đoán Viêm Phổi Qua Ảnh X-Quang Ngực | ResNet50 Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.markdown("#### 📁 Tải Ảnh X-Quang")
    
    uploaded_file = st.file_uploader(
        " ",  # Label trống, dùng CSS để hiển thị
        type=['jpg', 'png', 'jpeg'],
        help="Hỗ trợ: JPG, PNG, JPEG (tối đa 200MB)",
        label_visibility="collapsed"
    )
    
    # Sample images
    st.markdown("**Hoặc dùng ảnh mẫu:**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Bình thường", use_container_width=True):
            st.session_state['sample_image'] = 'data_samples/NORMAL/NORMAL_test_0000.jpeg'
            st.session_state['sample_cleared'] = False
            st.rerun()
    with col_s2:
        if st.button("Viêm phổi", use_container_width=True):
            st.session_state['sample_image'] = 'data_samples/PNEUMONIA/PNEUMONIA_test_0000.jpeg'
            st.session_state['sample_cleared'] = False
            st.rerun()
    
    # Clear sample when new file uploaded
    if uploaded_file is not None:
        st.session_state['sample_cleared'] = True
    
    # Handle image
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    elif 'sample_image' in st.session_state and not st.session_state.get('sample_cleared', False):
        if os.path.exists(st.session_state['sample_image']):
            image = Image.open(st.session_state['sample_image']).convert('RGB')
    
    if image:
        st.image(image, caption="Ảnh X-quang đã chọn", use_container_width=True)
        
        # Advanced settings (minimized)
        with st.expander("⚙️ Cài đặt nâng cao", expanded=False):
            alpha = st.slider("Độ đậm heatmap", 0.3, 0.7, 0.5, 0.05)
            show_probabilities = st.checkbox("Hiển thị xác suất dự đoán", value=True)
        
        if 'alpha' not in locals():
            alpha = 0.5
            show_probabilities = True
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Phân Tích", type="primary", use_container_width=True)
    else:
        analyze_btn = False
        alpha = 0.5
        show_probabilities = True

if image and analyze_btn and model:
    with col_right:
        st.markdown("#### 📊 Kết Quả Phân Tích")
        
        with st.spinner("Đang phân tích ảnh X-quang..."):
            try:
                # A. Prediction
                input_tensor = process_image(image)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                pred_class = pred.item()
                confidence = conf.item() * 100
                prob_normal = probs[0][0].item() * 100
                prob_pneumonia = probs[0][1].item() * 100
                
                # B. Display result
                if pred_class == 1:  # PNEUMONIA
                    st.markdown(f"""
                    <div class='result-positive'>
                        <h3 style='color: #991b1b; margin: 0;'>⚠️ Phát Hiện Viêm Phổi</h3>
                        <p style='color: #7f1d1d; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                            Độ tin cậy: <strong>{confidence:.1f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probabilities
                    if show_probabilities:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Bình thường", f"{prob_normal:.1f}%")
                        with col2:
                            st.metric("Viêm phổi", f"{prob_pneumonia:.1f}%")
                    
                    # Generate Grad-CAM ONLY for pneumonia
                    st.markdown("**🔬 Vùng Tổn Thương AI Phát Hiện:**")
                    with st.spinner("Đang tạo bản đồ nhiệt..."):
                        grad_img = run_gradcam_advanced(model, image, alpha)
                        
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(image, caption="Ảnh gốc", use_container_width=True)
                        with col_img2:
                            st.image(grad_img, caption="Vùng AI chú ý (màu đỏ/cam)", use_container_width=True)
                    
                    st.info("""
**🔬 Grad-CAM Heatmap - Vùng Tổn Thương Nghi Ngờ**

• Vùng đỏ/cam: Khu vực có đặc điểm hình ảnh học tương đồng với viêm phổi (opacity, infiltrate)

• Sử dụng: Công cụ hỗ trợ second opinion, giúp rút ngắn thời gian screening

• Khuyến nghị: Đánh giá kết hợp với triệu chứng lâm sàng, tiền sử bệnh, và các xét nghiệm khác
                    """)
                    
                else:  # NORMAL
                    st.markdown(f"""
                    <div class='result-negative'>
                        <h3 style='color: #065f46; margin: 0;'>✓ Không Phát Hiện Viêm Phổi</h3>
                        <p style='color: #064e3b; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                            Độ tin cậy: <strong>{confidence:.1f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probabilities
                    if show_probabilities:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Bình thường", f"{prob_normal:.1f}%")
                        with col2:
                            st.metric("Viêm phổi", f"{prob_pneumonia:.1f}%")
                        
                        # Debug info
                        with st.expander("🔍 Thông tin debug (cho developer)"):
                            st.code(f"Raw outputs: {outputs[0].tolist()}")
                            st.code(f"Softmax probs: [Normal={prob_normal:.2f}%, Pneumonia={prob_pneumonia:.2f}%]")
                            st.code(f"Predicted class: {pred_class} ({'Pneumonia' if pred_class==1 else 'Normal'})")
                    
                    # NO Grad-CAM for normal cases
                    st.image(image, caption="Ảnh X-quang", use_container_width=True)
                    
                    st.markdown("""
                    <div class='info-card'>
                        <p style='color: #475569; margin: 0; line-height: 1.7;'>
                            <strong style='color: #0066cc;'>📋 Đánh Giá:</strong><br>
                            • AI không phát hiện dấu hiệu viêm phổi điển hình<br>
                            • Kết quả này có thể tham khảo kết hợp triệu chứng lâm sàng<br>
                            • Viêm phổi giai đoạn sớm có thể chưa rõ ràng trên X-quang
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Disclaimer
                st.markdown("---")
                st.caption("💡 **Vai trò AI:** Công cụ hỗ trợ quyết định lâm sàng (Clinical Decision Support System). Rút ngắn thời gian screening, second opinion tự động, giảm tải cho bác sĩ trong khối lượng ảnh lớn. Quyết định chẩn đoán cuối cùng dựa trên đánh giá tổng hợp của bác sĩ.")
                    
            except Exception as e:
                st.error(f"❌ Lỗi phân tích: {str(e)}")
                st.info("Vui lòng thử lại với ảnh khác hoặc kiểm tra cấu hình model.")

elif not model:
    with col_right:
        st.error("❌ Không thể tải model")
        st.info("Vui lòng đảm bảo file `final_pneumonia_model.pth` có trong thư mục `models/`")
elif image and not analyze_btn:
    with col_right:
        st.markdown("""
        <div class='info-card'>
            <p style='color: #64748b; margin: 0;'>
                👈 Nhấn nút <strong>Phân Tích</strong> để bắt đầu chẩn đoán bằng AI
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    with col_right:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #0066cc; margin-top: 0;'>Hướng Dẫn Sử Dụng</h4>
            <ol style='color: #475569; line-height: 1.8;'>
                <li>Tải lên ảnh X-quang ngực (JPG/PNG)</li>
                <li>Nhấn nút "Phân Tích" để AI đánh giá</li>
                <li>Xem kết quả và khuyến nghị lâm sàng</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
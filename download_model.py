import os
import gdown

def download_model():
    """
    Tự động tải model từ Google Drive nếu chưa có
    """
    model_path = 'models/final_pneumonia_model.pth'
    
    # Nếu model đã tồn tại, không cần tải
    if os.path.exists(model_path):
        print(f"✓ Model đã tồn tại: {model_path}")
        return model_path
    
    # Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)
    
    # Google Drive File ID (thay bằng ID thực tế)
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print("📥 Đang tải model từ Google Drive...")
    print("⏳ Vui lòng đợi, file khoảng 90MB...")
    
    try:
        gdown.download(url, model_path, quiet=False)
        print(f"✅ Đã tải model thành công: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        raise

if __name__ == "__main__":
    download_model()

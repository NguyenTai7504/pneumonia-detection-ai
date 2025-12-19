"""
Script tự động tải ảnh X-quang mẫu để test ứng dụng AI phát hiện viêm phổi
Nguồn: Public medical datasets
Lưu ý: Các ảnh được lấy từ nguồn công khai phục vụ mục đích demo
"""

import os
import requests
from pathlib import Path
import time

# Cấu hình
NORMAL_DIR = "data_samples/NORMAL"
PNEUMONIA_DIR = "data_samples/PNEUMONIA"
NUM_SAMPLES_PER_CLASS = 7  # 7 ảnh mỗi loại = 14 ảnh tổng

# URLs ảnh mẫu từ GitHub Dataset công khai
# ⚠️ Lưu ý: Đã kiểm tra và loại bỏ ảnh bị trùng lặp
SAMPLE_URLS = {
    "NORMAL": [
        # Ảnh phổi bình thường từ các nguồn y khoa
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0929664620300449-gr2_lrg-a.jpg",
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0929664620300449-gr2_lrg-b.jpg",
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/nejmc2001573_f1a.jpeg",
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/nejmc2001573_f1b.jpeg",
        "https://prod-images-static.radiopaedia.org/images/53396551/0001_gallery.jpeg",
        "https://prod-images-static.radiopaedia.org/images/53396550/0002_gallery.jpeg",
        "https://prod-images-static.radiopaedia.org/images/53396549/0003_gallery.jpeg",
    ],
    "PNEUMONIA": [
        # Ảnh viêm phổi rõ ràng (KHÔNG trùng với NORMAL)
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/streptococcus-pneumoniae-pneumonia-temporal-evolution-1-day0.jpg",
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg",
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/covid-19-pneumonia-7-PA.jpg",
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0140673620303706-fx1_lrg.jpg",
        "https://prod-images-static.radiopaedia.org/images/52166505/0001_gallery.jpeg",
        "https://prod-images-static.radiopaedia.org/images/52166506/0002_gallery.jpeg",
        "https://prod-images-static.radiopaedia.org/images/52166507/0003_gallery.jpeg",
    ]
}

def download_image(url, save_path, timeout=30):
    """Tải một ảnh từ URL"""
    try:
        print(f"  Đang tải: {url.split('/')[-1][:50]}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Lưu file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  ✅ Đã lưu: {save_path}")
        return True
    except Exception as e:
        print(f"  ❌ Lỗi tải {url}: {e}")
        return False

def main():
    print("=" * 60)
    print("SCRIPT TỰ ĐỘNG TẢI ẢNH X-QUANG MẪU")
    print("=" * 60)
    
    # Tạo thư mục nếu chưa có
    os.makedirs(NORMAL_DIR, exist_ok=True)
    os.makedirs(PNEUMONIA_DIR, exist_ok=True)
    print(f"✅ Đã tạo thư mục: {NORMAL_DIR}, {PNEUMONIA_DIR}\n")
    
    total_downloaded = 0
    total_failed = 0
    
    # Tải ảnh NORMAL
    print(f"📥 Đang tải {len(SAMPLE_URLS['NORMAL'])} ảnh BÌNH THƯỜNG...")
    for idx, url in enumerate(SAMPLE_URLS['NORMAL'][:NUM_SAMPLES_PER_CLASS], 1):
        # Lấy extension từ URL
        ext = url.split('.')[-1].split('?')[0]
        if ext not in ['jpg', 'jpeg', 'png']:
            ext = 'jpg'
        
        save_path = os.path.join(NORMAL_DIR, f"normal_{idx:03d}.{ext}")
        
        if download_image(url, save_path):
            total_downloaded += 1
        else:
            total_failed += 1
        
        time.sleep(0.5)  # Delay tránh spam
    
    print()
    
    # Tải ảnh PNEUMONIA
    print(f"📥 Đang tải {len(SAMPLE_URLS['PNEUMONIA'])} ảnh VIÊM PHỔI...")
    for idx, url in enumerate(SAMPLE_URLS['PNEUMONIA'][:NUM_SAMPLES_PER_CLASS], 1):
        ext = url.split('.')[-1].split('?')[0]
        if ext not in ['jpg', 'jpeg', 'png']:
            ext = 'jpg'
        
        save_path = os.path.join(PNEUMONIA_DIR, f"pneumonia_{idx:03d}.{ext}")
        
        if download_image(url, save_path):
            total_downloaded += 1
        else:
            total_failed += 1
        
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print(f"HOÀN THÀNH!")
    print(f"✅ Đã tải thành công: {total_downloaded} ảnh")
    print(f"❌ Thất bại: {total_failed} ảnh")
    print("=" * 60)
    
    # Tạo ảnh test mặc định cho main.py
    normal_files = list(Path(NORMAL_DIR).glob("*"))
    pneumonia_files = list(Path(PNEUMONIA_DIR).glob("*"))
    
    if pneumonia_files:
        test_image = pneumonia_files[0]
        print(f"\n💡 Ảnh test mặc định cho main.py: {test_image}")
        print(f"   Cập nhật dòng IMAGE_PATH trong main.py thành:")
        print(f"   IMAGE_PATH = '{test_image}'")

if __name__ == "__main__":
    main()

from ultralytics import YOLO
import torch

def start_training():
    # Cihaz kontrolü
    if torch.cuda.is_available():
        print(f"[DONANIM]: NVIDIA GPU Tespit Edildi: {torch.cuda.get_device_name(0)}")
        print("[DURUM]: RTX 3060 motorları ateşleniyor...")
    else:
        print("[UYARI]: GPU bulunamadı veya PyTorch GPU sürümü yüklü değil! CPU kullanılacak.")

    # Medium modeli yüklüyoruz (Daha yüksek başarı oranı)
    model = YOLO('yolov8m.pt') 

    print("[SİSTEM]: Nöral ağ eğitimi başlatılıyor (High Performance Mode)...")
    
    # Eğitimi Başlat
    model.train(
        data='config.yaml', 
        epochs=100,         # Medium model için tur sayısını artırdık, daha iyi öğrensin.
        imgsz=640,          # 640 standarttır, dokuları iyi görür.
        batch=16,           # 12GB VRAM için 16 veya 24 idealdir. Hata alırsan 16'ya düşür.
        name='burn_result_medium', # Klasör adı karışmasın
        device=0,           # 0 demek ilk GPU (RTX 3060) demektir.
        workers=8,          # Veri yüklemeyi hızlandırır.
        optimizer='AdamW',  # Daha modern bir optimizasyon algoritması (Opsiyonel ama iyidir)
        lr0=0.01            # Öğrenme oranı (Varsayılan iyidir ama aklında olsun)
    )
    
    print("[SİSTEM]: Görev tamamlandı. Yeni zeka şurada: runs/detect/burn_result_medium/weights/best.pt")

if __name__ == '__main__':
    start_training()
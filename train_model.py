from ultralytics import YOLO
import torch

def start_training():
    # Medium model
    model = YOLO('yolov8m.pt') 

    print("[SİSTEM]: Nöral ağ eğitimi başlatılıyor.")
    
    # Eğitimi Başlat
    model.train(
        data='config.yaml', 
        epochs=100,         # Medium model için tur sayısını artırdık, daha iyi öğrensin.
        imgsz=640,          # 640 standarttır, dokuları iyi görür.
        batch=16,           # 12GB VRAM için 16 veya 24 idealdir. Hata alırsan 16'ya düşür.
        name='burn_result_medium', # Klasör adı karışmasın
        device=0,           # 0 demek ilk GPU demek.
        workers=8,          # Veri yüklemeyi hızlandırır.
        optimizer='AdamW',  # optimizasyon algoritması
        lr0=0.01            # Öğrenme oranı
    )
    
    print("[SİSTEM]: Görev tamamlandı. Yeni zeka şurada: runs/detect/burn_result_medium/weights/best.pt")

if __name__ == '__main__':
    start_training()

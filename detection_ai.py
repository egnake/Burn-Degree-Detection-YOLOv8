import cv2
from ultralytics import YOLO
import math
import time

# --- AYARLAR ---
# Eğitim bitince oluşan dosyanın yolu (Burası sende farklı olabilir, kontrol et!)
MODEL_PATH = 'runs/detect/burn_result/weights/best.pt' 

# Renk Paleti (BGR Formatında)
COLOR_1ST = (0, 255, 255)   # Sarı
COLOR_2ND = (0, 128, 255)   # Turuncu
COLOR_3RD = (0, 0, 255)     # Kırmızı
COLOR_TXT = (255, 255, 255) # Beyaz

# --- TAVSİYE DATABASE ---
def get_medical_advice(degree):
    if "1." in degree:
        return [
            "DURUM: Yuzeysel hasar (Epidermis).",
            "OK: 10-20 dk soguk su (buz degil).",
            "OK: Nemlendirici / Aloe Vera.",
            "YASAK: Dis macunu, yogurt vb."
        ]
    elif "2." in degree:
        return [
            "DURUM: Derin hasar, su toplamasi.",
            "OK: Steril bezle kapat.",
            "OK: Agri kesici.",
            "YASAK: Baloncuklari patlatma!"
        ]
    elif "3." in degree:
        return [
            "KRITIK: Sinir hasari, his kaybi.",
            "ACIL: Hemen 112'yi ara.",
            "OK: Yaraliyi temiz ortuyle sar.",
            "YASAK: Kiyafetleri cikarma, su surme!"
        ]
    return ["Tanimlanamayan durum."]

def main():
    print("[INIT] Sistem yukleniyor...")
    try:
        model = YOLO(MODEL_PATH)
    except:
        print(f"HATA: Model dosyasi '{MODEL_PATH}' bulunamadi. Once egitim yapmalisin!")
        return

    # Kamera secimi (0 genellikle laptop kamerasidir)
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # Genislik
    cap.set(4, 720)  # Yukseklik

    paused = False # Dondurma modu kontrolü
    frame_to_show = None

    print("[READY] Sistem hazir. Cikmak icin 'q', Dondurmak/Analiz icin 'SPACE'.")

    while True:
        if not paused:
            success, img = cap.read()
            if not success: break
            
            # Yapay Zeka Tahmini
            results = model(img, stream=True, verbose=False)
            
            # Tespitleri Ciz
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Kordinatlar
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Sinif ve Guven
                    cls = int(box.cls[0])
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    name = model.names[cls]

                    # Renge karar ver
                    if cls == 0: color = COLOR_1ST
                    elif cls == 1: color = COLOR_2ND
                    else: color = COLOR_3RD

                    # Kutuyu Ciz
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Etiketi Yaz (Modern Gorunum)
                    label = f"{name} %{int(conf*100)}"
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # Yazı arka planı
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            # Ekrana HUD bilgileri ekle
            cv2.putText(img, "CANLI ANALIZ MODU", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(img, "[SPACE]: Dondur/Detay  [Q]: Cikis", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
            
            frame_to_show = img

        else:
            # --- DONDURULMUŞ EKRAN (ANALİZ MODU) ---
            # Görüntüyü hafif karart (Blur efekti gibi)
            overlay = frame_to_show.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 720), (0, 0, 0), -1)
            frame_to_show = cv2.addWeighted(overlay, 0.7, frame_to_show, 0.3, 0)
            
            # Yan panel ciz
            cv2.rectangle(frame_to_show, (800, 0), (1280, 720), (20, 20, 20), -1)
            cv2.putText(frame_to_show, "DETAYLI ANALIZ RAPORU", (820, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Son tespit edilen sınıfa göre tavsiye ver (Basitlik icin son kutuyu aliyoruz)
            # Gerçek senaryoda ekrandaki en büyük yanığı baz almak gerekir.
            try:
                if 'name' in locals(): # Eğer bir tespit varsa
                    advice_list = get_medical_advice(name)
                    y_pos = 100
                    for line in advice_list:
                        color_line = (0, 255, 0) if "OK" in line else (0, 0, 255) if "YASAK" in line or "ACIL" in line else (255, 255, 255)
                        cv2.putText(frame_to_show, line, (820, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_line, 1)
                        y_pos += 40
                else:
                     cv2.putText(frame_to_show, "Yanik tespit edilemedi.", (820, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            except:
                pass

            cv2.putText(frame_to_show, "DEVAM ETMEK ICIN 'SPACE'", (820, 650), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)

        cv2.imshow('Burn Sentinel AI', frame_to_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '): # Space tuşu
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
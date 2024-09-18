import os
from ultralytics import YOLO

# Path ke file data.yaml
# yaml_file_path = 'Augmented2064/data.yaml'  # Sesuaikan path ini dengan letak file data.yaml Anda

# Load YOLOv8 model (Nano version). Anda bisa mengganti 'yolov8n.pt' dengan model lain seperti 'yolov8s.pt' untuk Small model
model = YOLO('yolov8n.pt')

# Melakukan training menggunakan dataset yang sudah didefinisikan dalam file data.yaml
model.train(
    data='data2064.yaml',  # Path ke file data.yaml
    epochs=10,            # Jumlah epoch training
    imgsz=[2048, 1536],            # Ukuran gambar (rescale ke 640x640 piksel)
    batch=1              # Ukuran batch
)

# Evaluasi model pada validation set
metrics = model.val(data='data2064.yaml')

# Prediksi pada test set dan simpan hasilnya
results = model.predict(source='D:/RecycloRobo_ai/Augmented2064/test/images', save=True, save_dir='D:/RecycloRobo_ai/Augmented2064/predictions')

# Menampilkan hasil prediksi (bounding boxes dan labels)
# results.show()

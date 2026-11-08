# TASK 03 - Preprocessing and Augmentation

## 1. Tujuan TASK 03

TASK 03 menyediakan preprocessing dan augmentasi image classification yang modular, model-aware, dapat direproduksi, dan tidak mengubah split aktif. Implementasi ini menyiapkan kontrak reusable untuk notebook training berikutnya tanpa menjalankan training, inference skala penuh, atau evaluasi model.

## 2. Kondisi Dataset Sebelum Preprocessing

Dataset aktif TASK 02E terdiri dari 1.914 gambar V1/train: 1.340 train, 383 validation, dan 191 internal test. Delapan kelas diperoleh dari split manifest: cardboard, electronic, foodwaste, glass, metal, paper, plastic, dan textile. Dua record ND-002 tetap tidak ada pada dataset aktif karena label noise terkonfirmasi.

## 3. Arsitektur Modular

`src/data/preprocessing.py` menangani decoding RGB, resize, crop/padding, tensor conversion, scaling, normalization, validasi tensor, resolusi konfigurasi, dan transform metadata pretrained. `src/data/augmentation.py` menangani augmentasi berbasis citra pada train, adapter Ultralytics, dan preview. Pemisahan ini mencegah augmentasi diterapkan sesudah normalization atau pada split evaluasi.

## 4. Perbedaan Preprocessing dan Augmentasi

Preprocessing adalah transformasi input yang diperlukan model dan dapat deterministik: RGB, resize, tensor, scaling, dan normalization. Augmentasi adalah variasi acak image-domain untuk train seperti flip horizontal, rotasi, shear, dan brightness terbatas. Augmentasi tidak pernah menyimpan file baru atau memodifikasi label.

## 5. Baseline Resize 224 x 224

CNN from scratch memakai baseline 224 x 224 untuk kompatibilitas eksperimen awal, biaya memori yang sesuai GPU 4 GB, dan kesetaraan awal dengan banyak arsitektur pretrained. Nilai ini adalah konfigurasi, bukan asumsi global; profil model lain dapat memakai metadata bobot resmi atau override tervalidasi.

## 6. Dukungan Model-Specific

`cnn.yaml` memakai direct resize dan scaling [0, 1] tanpa normalization wajib. MobileNetV3 dan ResNet50 memakai profil torchvision fallback serta dapat diprioritaskan oleh `weights.transforms()`. EfficientNet memakai profil timm fallback serta dapat diprioritaskan oleh `resolve_model_data_config()` dan `create_transform()`. YOLOv8 memakai adapter parameter Ultralytics dan tidak menggunakan transform torchvision/timm eksternal.

## 7. Prioritas Metadata Pretrained

Jika caller memberikan torchvision weights, `build_transforms()` memakai `weights.transforms()` dan tidak menambahkan resize atau normalization konfigurasi kedua. Jika caller memberikan model timm, metadata `resolve_model_data_config(model=model)` dan `create_transform(..., is_training=False)` dipakai. Konfigurasi YAML menjadi fallback bila metadata atau objek bobot/model tidak diberikan. Override eksplisit harus ditaruh pada `preprocessing.overrides` dan tetap melewati validasi.

## 8. Strategi Normalization

CNN baseline hanya memakai `ToTensor` dengan rentang [0, 1]. Fallback pretrained memakai statistik ImageNet pada YAML. Metadata resmi pretrained mengatur normalization sendiri dan mencegah normalization ganda. Validator memastikan tensor float32, CHW, finite, dan berada di [0, 1] bila tidak dinormalisasi.

## 9. Jenis dan Parameter Augmentasi

Profil `cnn_standard` memakai horizontal flip p=0,5; rotasi maksimum 15 derajat; shear horizontal/vertikal maksimum 10 derajat; dan brightness maksimum 10%. `RandomBrightnessContrast` digunakan dengan contrast nol sebagai pendekatan brightness; transform ini bukan simulasi exposure fisik yang ekuivalen. Vertical flip tidak digunakan. Profil pretrained lebih ringan, sedangkan YOLO memetakan flip, degrees, dan shear ke parameter trainer Ultralytics.

## 10. Urutan Transformasi

Pada train, input didekode ke RGB, augmentasi image-domain dilakukan, lalu preprocessing model mengubah image menjadi tensor dan menormalisasi jika perlu. Pada validation, internal test, dan inference, augmentasi dihapus sehingga jalur menjadi RGB lalu preprocessing deterministik. Urutan ini menghindari augmentasi terhadap tensor yang sudah dinormalisasi.

## 11. YAML Configuration

`configs/preprocessing.yaml` menyimpan profil resize/scaling/normalization. `configs/augmentation.yaml` menyimpan probabilitas dan batas augmentasi. Profil model tersimpan terpisah pada `configs/models/`. Validator menolak ukuran invalid, interpolation tidak dikenal, std nol, normalization dengan scaling tidak sesuai, probabilitas di luar [0, 1], dan vertical flip baseline YOLO.

## 12. API yang Disediakan

API utama adalah `build_transforms(model_type, split, model=None, weights=None, config=None, project_root=None, augmentation=None)`. `build_training_augmentation(profile_name, enabled=True, seed=42)` membuat augmentasi train on-the-fly. `build_yolo_classification_adapter()` menghasilkan dictionary parameter Ultralytics tanpa transform eksternal. `load_active_class_mapping()` membaca mapping kelas konsisten dari manifest aktif.

## 13. Contoh Integrasi Model

CNN dapat memakai `build_transforms("cnn", "train", augmentation=build_training_augmentation("cnn_standard"))`. MobileNetV3 atau ResNet50 dapat memberikan enum weights torchvision pada parameter `weights`. EfficientNet timm dapat memberikan instance model pada parameter `model`; implementasi tidak membuat atau mengunduh model. YOLO memakai `build_yolo_classification_adapter()` untuk meneruskan `imgsz`, `fliplr`, `degrees`, dan `shear` kepada trainer Ultralytics di notebook nanti.

## 14. Reproducibility

Seed default 42 digunakan saat membangun pipeline Albumentations dan preview. Dataset split tidak disentuh oleh transformasi; transformasi random hanya hidup di memori ketika sample train diambil. Validation dan internal test memakai transform tanpa augmentasi sehingga pemanggilan ulang untuk input yang sama deterministik.

## 15. Pencegahan Data Leakage

TASK 03 hanya membaca `data/processed/` aktif dan manifest aktif untuk preview. Tidak ada kode yang mengakses V5 atau V1/test. ND-002 tidak dapat dipilih karena tidak ada pada split manifest atau direktori aktif. Tidak ada statistik dataset yang dipelajari dari validation/internal test, tidak ada pembagian ulang, dan tidak ada penulisan ke `data/processed/`.

## 16. Augmentation Preview

`outputs/figures/augmentation_preview.png` dibuat dari satu gambar aktual pada split train yang dipilih dari manifest aktif. Panel menampilkan original, preprocessing CNN 224 x 224, horizontal flip, rotation, shear, brightness approximation, dan kombinasi train. Preview preprocessing CNN tidak dinormalisasi sehingga dapat ditampilkan sebagai RGB yang benar.

## 17. Hasil Unit Test Aktual

`tests/test_preprocessing.py` menghasilkan 7 test lulus. Test mencakup RGB conversion, resize, CHW, float32, finite tensor, normalization, penolakan konfigurasi normalization tidak valid, determinisme validation/internal test, train augmentation enable/disable, profil independen, metadata torchvision weights, adapter YOLO tanpa augmentasi ganda, invalid image, serta ketidakberubahan manifest dan file sumber.

## 18. Keterbatasan Implementasi

Integrasi torchvision diuji pada metadata enum tanpa download bobot dan tanpa training. Integrasi timm disediakan sebagai jalur metadata ketika instance model tersedia, tetapi tidak diuji dengan model nyata agar TASK 03 tidak menginstansiasi atau mengunduh model. Parameter Ultralytics dipetakan dari konfigurasi versi lokal; `hsv_v` dibiarkan null karena HSV value jitter bukan exposure-equivalent.

## 19. Keputusan Teknis dan Alasannya

Albumentations dipakai hanya untuk image-domain augmentation sehingga resize/normalization tidak terjadi dua kali. Transform pretrained resmi diprioritaskan karena ukuran resize, interpolation, crop, dan normalization dapat berbeda antar-bobot. YOLO dipisahkan karena pipeline classification Ultralytics memiliki pemilik transformasi sendiri. Semua konfigurasi berada pada YAML agar eksperimen notebook dapat berubah tanpa mengedit kode bersama.

## 20. Rekomendasi untuk TASK 04

TASK 04 sebaiknya membangun dataloader notebook di atas `split_manifest.csv`, `load_active_class_mapping()`, dan API transform ini. Catat model profile, weights identifier, seed, batch size, serta override konfigurasi pada setiap eksperimen. Gunakan train untuk augmentasi dan validation/internal test tanpa augmentasi; jangan mengubah split aktif atau memasukkan kembali ND-002 sebelum kebijakan label baru tersedia.

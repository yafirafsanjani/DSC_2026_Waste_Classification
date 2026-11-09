# Dokumentasi DSC 2026 Waste Classification

## Indeks

Dokumentasi ini merekam keputusan data pipeline sebelum tahap preprocessing, augmentasi, dan training. Seluruh proses menggunakan gambar asli secara read-only; hasil split di `data/processed/` adalah salinan terverifikasi dari sumber yang diizinkan.

1. [TASK 01 - Dataset Audit](TASK_01_Dataset_Audit.md)
2. [TASK 02 - Cleaning and Splitting](TASK_02_Cleaning_and_Splitting.md)
3. [TASK 02B - Near-Duplicate Review](TASK_02B_Near_Duplicate_Review.md)
4. [TASK 02C - Leakage-Safe Resplitting](TASK_02C_Leakage_Safe_Resplitting.md)
5. [TASK 02D - Final Manual Review](TASK_02D_Final_Manual_Review.md)
6. [TASK 02E - Final Dataset Resolution](TASK_02E_Final_Dataset_Resolution.md)
7. [TASK 03 - Preprocessing and Augmentation](TASK_03_Preprocessing_and_Augmentation.md)
8. [TASK 04 - Training Utilities and Callbacks](TASK_04_Training_Utilities_and_Callbacks.md)
9. [TASK 05B - CNN From Scratch Final Results](TASK_05_CNN_From_Scratch.md)
10. [TASK 05C - Test Inference and Kaggle Submission](TASK_05C_Test_Inference_Kaggle_Submission.md)
11. [TASK 05D - Standalone CNN Inference Fix](TASK_05D_Standalone_Inference_Fix.md)
12. [Standard - Training Monitoring and Kaggle Submission](STANDARD_Training_Monitoring_and_Kaggle_Submission.md)
13. [TASK 06A - CNN V2 Universal Monitoring](TASK_06A_CNN_V2_Universal_Monitoring.md)

## Status Data Saat Ini

- Dataset aktif tetap berisi 1.914 gambar dari V1/train: train 1.340, validation 383, dan internal test 191.
- TASK 03 hanya menambah transformasi on-the-fly, konfigurasi, preview, serta test. Manifest, gambar sumber, label, dan direktori split tidak diubah.
- Augmentasi acak hanya disiapkan untuk train. Validation, internal test, dan inference standar menggunakan transform deterministik.
- `v1/test` dan seluruh `v5` tetap dilarang untuk preprocessing eksperimen, training, evaluasi, dan pemilihan model.
- TASK 04 hanya menyiapkan DataLoader manifest-backed, loop, callbacks, checkpoint, resume, monitoring, dan pengujian sintetis. Tidak ada model kompetisi yang dilatih atau dievaluasi.
- TASK 05B mendokumentasikan hasil aktual baseline CNN from scratch: 75 epoch dari maksimum 100, early stopping pada `val_loss`, dan best checkpoint pada epoch 64 dengan validation loss 1.248915, accuracy 59.01%, serta macro F1 58.69%. Internal test tetap tidak digunakan untuk pemilihan model.
- TASK 05C menambahkan audit test resmi 480 gambar, dataset inference unlabeled yang hanya memakai `data/competition/test/`, loader best checkpoint CNN, dan generator CSV Kaggle manual opt-in. Tidak ada inference penuh atau submission CSV yang dibuat otomatis.
- TASK 05D membuat setup inference CNN benar-benar mandiri untuk kernel Jupyter baru; root proyek, konfigurasi, mapping kelas, checkpoint, dan path dibaca dari artefak disk tanpa bergantung pada cell training.
- TASK 06A menerapkan standar monitoring universal pertama pada notebook CNN From Scratch V2: progress batch, ringkasan epoch, history JSON/CSV/JSONL yang aman, learning curves, serta workflow submission standalone. Model pretrained dan YOLO belum diimplementasikan.

## Reproduksi

Gunakan virtual environment `venv_gelarrasa` dan jalankan dari root proyek. Tidak ada perintah training dalam dokumentasi ini.

```powershell
.\venv_gelarrasa\Scripts\python.exe -m pytest tests\test_preprocessing.py -q
.\venv_gelarrasa\Scripts\python.exe -m pytest tests\test_dataset.py -q
```

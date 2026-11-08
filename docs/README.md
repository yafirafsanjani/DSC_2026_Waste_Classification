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
9. [TASK 05 - CNN From Scratch Experiment](TASK_05_CNN_From_Scratch.md)

## Status Data Saat Ini

- Dataset aktif tetap berisi 1.914 gambar dari V1/train: train 1.340, validation 383, dan internal test 191.
- TASK 03 hanya menambah transformasi on-the-fly, konfigurasi, preview, serta test. Manifest, gambar sumber, label, dan direktori split tidak diubah.
- Augmentasi acak hanya disiapkan untuk train. Validation, internal test, dan inference standar menggunakan transform deterministik.
- `v1/test` dan seluruh `v5` tetap dilarang untuk preprocessing eksperimen, training, evaluasi, dan pemilihan model.
- TASK 04 hanya menyiapkan DataLoader manifest-backed, loop, callbacks, checkpoint, resume, monitoring, dan pengujian sintetis. Tidak ada model kompetisi yang dilatih atau dievaluasi.
- TASK 05 menambahkan notebook baseline CNN from scratch dengan cell opt-in manual. Notebook belum menghasilkan metrik eksperimen dan tidak menggunakan internal test untuk pemilihan model.

## Reproduksi

Gunakan virtual environment `venv_gelarrasa` dan jalankan dari root proyek. Tidak ada perintah training dalam dokumentasi ini.

```powershell
.\venv_gelarrasa\Scripts\python.exe -m pytest tests\test_preprocessing.py -q
.\venv_gelarrasa\Scripts\python.exe -m pytest tests\test_dataset.py -q
```

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
14. [TASK 07 - EfficientNet-B2 Pretrained](TASK_07_EfficientNet_B2_Pretrained.md)
15. [TASK 08 - Unified Few-Shot EfficientNet-B2 & Pseudo-Labeling](TASK_08_Unified_FewShot_PseudoLabeling.md)
16. [TASK 09 - Optimized Few-Shot Classification & SSL V2](TASK_09_Optimized_FewShot_SSL.md)

## Status Data Saat Ini

- Dataset aktif tetap berisi 1.914 gambar dari V1/train: train 1.340, validation 383, dan internal test 191.
- TASK 03 hanya menambah transformasi on-the-fly, konfigurasi, preview, serta test. Manifest, gambar sumber, label, dan direktori split tidak diubah.
- Augmentasi acak hanya disiapkan untuk train. Validation, internal test, dan inference standar menggunakan transform deterministik.
- `v1/test` dan seluruh `v5` tetap dilarang untuk preprocessing eksperimen, training, evaluasi, dan pemilihan model.
- TASK 04 hanya menyiapkan DataLoader manifest-backed, loop, callbacks, checkpoint, resume, monitoring, dan pengujian sintetis. Tidak ada model kompetisi yang dilatih atau dievaluasi.
- TASK 05B mendokumentasikan hasil aktual baseline CNN from scratch: 75 epoch dari maksimum 100, early stopping pada `val_loss`, dan best checkpoint pada epoch 64 dengan validation loss 1.248915, accuracy 59.01%, serta macro F1 58.69%. Internal test tetap tidak digunakan untuk pemilihan model.
- TASK 05C menambahkan audit test resmi 480 gambar, dataset inference unlabeled yang hanya memakai `data/competition/test/`, loader best checkpoint CNN, dan generator CSV Kaggle manual opt-in. Tidak ada inference penuh atau submission CSV yang dibuat otomatis.
- TASK 05D membuat setup inference CNN benar-benar mandiri untuk kernel Jupyter baru; root proyek, konfigurasi, mapping kelas, checkpoint, dan path dibaca dari artefak disk tanpa bergantung pada cell training.
- TASK 06A menerapkan standar monitoring universal pertama pada notebook CNN From Scratch V2: progress batch, ringkasan epoch, history JSON/CSV/JSONL yang aman, learning curves, serta workflow submission standalone. Pada TASK 06A, model pretrained dan YOLO belum diimplementasikan; lihat TASK 07 untuk EfficientNet-B2.
- TASK 07: `TASK_07_EfficientNet_B2_Pretrained.md` mendokumentasikan EfficientNet-B2 pretrained dua fase pada dataset V1 lama; eksperimen ini merupakan pengujian teknis pra-migrasi, bukan bukti kepatuhan aturan lima label per kelas.
- TASK 08: `TASK_08_Unified_FewShot_PseudoLabeling.md` mendokumentasikan baseline semi-supervised awal (skor Kaggle 0.5902).
- TASK 09: `TASK_09_Optimized_FewShot_SSL.md` mengimplementasikan notebook optimasi `notebooks/09_optimized_fewshot_ssl.ipynb`: memperbaiki BatchNorm running stats leakage saat backbone dibekukan, mengevaluasi 4 classifier few-shot (PyTorch Linear, L2 Logistic Regression, Linear SVC, Cosine Prototype) via 5-Fold Stratified CV OOF, menerapkan two-view augmentation consistency pseudo-labeling, Class-Balanced Top-K filtering, Student training dengan weighted loss, Test-Time Augmentation (TTA), serta ekspor `submission_optimized_fewshot_ssl.csv`. Notebook telah direfaktor secara penuh untuk mendukung eksekusi **Kernel Restart & Run All** yang reproducible bagi juri/kompetisi, mengeliminasi seluruh 10 flag manual dan menggantinya dengan kamus `CONFIG` terpusat, membaca dataset resmi `data/competition/train/` (1.920 citra) dan `test/` (480 citra), memvalidasi kuota ketat 5 label/kelas (40 total) via `data/manifests/fewshot_pseudolabel_b2_v1/labeled_pool_40.csv`, mengarantina 3 citra train bertabrakan test, mengalokasikan 1.877 citra unlabeled pool bersih, 5-Fold Stratified Cross-Validation, Teacher-Student Pseudo-Labeling (threshold >= 0.95), weighted loss semi-supervised, evaluasi komparatif, inferensi test otomatis, serta pembuatan berkas `submission_fewshot_pseudolabel_b2.csv`.

## Reproduksi

Gunakan virtual environment `venv_gelarrasa` dan jalankan dari root proyek. Tidak ada perintah training dalam dokumentasi ini.

```powershell
.\venv_gelarrasa\Scripts\python.exe -m pytest tests\test_preprocessing.py -q
.\venv_gelarrasa\Scripts\python.exe -m pytest tests\test_dataset.py -q
.\venv_gelarrasa\Scripts\python.exe -m pytest tests\test_fewshot_pseudolabel_b2.py -v
.\venv_gelarrasa\Scripts\python.exe -m pytest tests\test_optimized_fewshot_ssl.py -v
```

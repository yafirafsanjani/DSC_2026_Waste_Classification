# TASK 05D - Standalone CNN Inference Fix

## 1. Tujuan

TASK 05D memperbaiki section **Test Inference & Kaggle Submission** agar dapat dijalankan dari kernel Jupyter baru tanpa menjalankan training, validation, resume, continued training, atau cell setup training sebelumnya. Perbaikan ini tetap memakai CNN checkpoint, preprocessing, class mapping, audit test, dan submission validator yang sama dengan TASK 05C.

Tidak ada training ulang, full inference 480 gambar, internal-test evaluation, atau CSV final otomatis yang dijalankan saat TASK ini.

## 2. Penyebab Error

`ModuleNotFoundError: No module named 'src'` terjadi karena cell inference TASK 05C langsung mengimpor `src.*`, tetapi pada kernel baru root proyek belum tentu ada di `sys.path`. Hal ini umum terjadi saat Jupyter membuka notebook dengan working directory `notebooks/`.

Setelah path diperbaiki secara manual, `NameError: name 'EXPERIMENT_ID' is not defined` terjadi karena section TASK 05C masih bergantung pada state cell training sebelumnya. Dependensi tersembunyi tersebut adalah `PROJECT_ROOT`, `EXPERIMENT_ID`, `CONFIG`, `DEVICE`, `pd`, `display`, serta asumsi bahwa class mapping dan konfigurasi telah tersimpan di memori kernel.

## 3. Perubahan Implementasi

Section inference di `notebooks/01_cnn_from_scratch.ipynb` sekarang memiliki tiga cell berurutan:

1. **Standalone Inference Setup** mengimpor library sendiri, menemukan root proyek berdasarkan struktur `src/`, `configs/`, dan `data/`, menambahkan root ke `sys.path`, lalu mengimpor modul proyek.
2. **Test Inference & Submission** memiliki `RUN_TEST_INFERENCE = False` dan hanya menjalankan prediksi bila pengguna mengubahnya menjadi `True`.
3. **Optional CSV Verification** hanya membaca serta memvalidasi ulang CSV yang sudah ada; cell ini tidak melakukan inference atau training.

Setup mendefinisikan `INFERENCE_EXPERIMENT_ID = "cnn_scratch_baseline_v1"`, lalu membaca `outputs/metrics/cnn_scratch_baseline_v1_summary.json`, `configs/models/cnn.yaml`, manifest class mapping aktif, dan metadata `best.pt`. Batch size (16), dropout, class mapping, dan checkpoint berasal dari artefak eksperimen yang telah direkam, bukan dari variabel training lama.

`src/inference/competition.py`, `src/inference/predict.py`, dan `src/inference/submission.py` sudah tidak membutuhkan notebook state: seluruh root/path/context diterima eksplisit saat dipanggil. `src/models/cnn_from_scratch.py` menyediakan arsitektur CNN kanonis yang sama dengan checkpoint.

## 4. Validasi Preprocessing and Checkpoint

Setup membangun transform `val` dan `inference` menggunakan konfigurasi model `cnn`, lalu memastikan keduanya identik dan tidak memiliki augmentasi. Konfigurasi yang diverifikasi adalah RGB, direct resize 224 x 224, bilinear interpolation, CHW `float32`, scaling [0, 1], dan tanpa normalization tambahan.

Setup hanya memakai `models/checkpoints/cnn_scratch_baseline_v1/best.pt`. Metadata harus memiliki `model_identifier` `cnn_from_scratch`, delapan kelas kontigu, dan class mapping yang sama dengan manifest serta summary TASK 05B. Loading model pada cell prediksi memanggil `eval()` dan prediksi memakai `torch.inference_mode()`.

## 5. Cara Menjalankan dari Kernel Baru

1. Buka `notebooks/01_cnn_from_scratch.ipynb` dan pilih kernel `venv_gelarrasa`.
2. Jalankan **Cell 1 - Standalone Inference Setup** di section Test Inference & Kaggle Submission. Cell ini melakukan audit read-only 480 gambar dan mengecek checkpoint; cell tidak membuat CSV.
3. Pastikan audit menampilkan 480 gambar test dan 480 baris sample submission tanpa error.
4. Ubah `RUN_TEST_INFERENCE = False` menjadi `True` pada **Cell 2 - Test Inference & Submission**.
5. Jalankan Cell 2 satu kali untuk membuat dan memvalidasi submission CNN.
6. Opsional: jalankan **Cell 3 - Optional CSV Verification** untuk membaca ulang file tanpa inference ulang.

Flag training tetap tidak berubah: `RUN_TRAINING = False`, `RUN_RESUME = False`, dan `RUN_CONTINUED_TRAINING = False`.

## 6. Test yang Dilakukan

- Setup dieksekusi dalam proses Python baru dari working directory `notebooks/` setelah root proyek dihapus dari `sys.path`; setup berhasil menambahkan root sendiri dan tidak menghasilkan `ModuleNotFoundError` atau `NameError`.
- Audit modul mencatat 480/480 gambar test dapat dibuka, sample submission memiliki 480 ID, dan preprocessing inference sama dengan validation.
- Smoke test memakai satu gambar resmi (`001`) tanpa menulis CSV: tensor berukuran `(3, 224, 224)`, forward pass menghasilkan logits `(1, 8)`, model berada pada mode eval, dan tidak ada gradient parameter.
- SHA-256 `best.pt` sebelum dan sesudah setup/smoke test identik.
- `pytest tests/test_competition_inference.py -q` lulus: 8 test.

## 7. CSV and Safety Controls

Output CNN hanya ditulis ke `Final Dataset for Submission to Kaggle/submission_cnn_from_scratch.csv`. Template submission hanya menentukan header dan urutan ID; nilai `waste_category` template tidak digunakan sebagai jawaban.

Writer mempertahankan leading zero pada `image_id`, menolak kategori di luar delapan kelas, memverifikasi urutan dan jumlah 480 ID, serta memakai `overwrite=False`. CSV model lain tidak disentuh. Tidak ada submission final dibuat oleh TASK 05D.

## 8. Batasan

- Setup mandiri membutuhkan artefak lokal yang tetap ada: konfigurasi, summary JSON, manifest aktif, checkpoint `best.pt`, test resmi, dan sample submission.
- Audit berhenti dengan error jika header CSV, ID, jumlah gambar, atau struktur test berubah; pengguna harus memperbaiki input sebelum inference.
- Hasil submission tetap bergantung pada kualitas CNN baseline. TASK ini hanya memperbaiki reproducibility dan keselamatan inference, bukan meningkatkan performa model.

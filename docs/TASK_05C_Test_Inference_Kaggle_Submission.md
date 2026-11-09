# TASK 05C - Test Inference and Kaggle Submission per Model

## 1. Tujuan

TASK 05C menambahkan pipeline inference reusable dan generator CSV submission yang terpisah dari training. Implementasi ini menyiapkan CNN From Scratch untuk inference manual pada test resmi kompetisi melalui best checkpoint, lalu menyimpan submission khusus model tanpa memakai label test atau nilai kategori sample submission sebagai prediksi.

TASK ini tidak menjalankan training, resume, fine-tuning, evaluasi validation ulang, evaluasi internal test, atau full inference 480 gambar. Full inference hanya terjadi jika pengguna secara eksplisit mengubah `RUN_TEST_INFERENCE` menjadi `True` di notebook.

## 2. Audit Data Resmi Kompetisi

Audit read-only pada `data/competition/test/` dan `data/competition/SampleSubmission_DSC.csv` selesai pada 9 November 2026.

| Pemeriksaan | Hasil |
| --- | --- |
| Lokasi satu-satunya untuk submission | `data/competition/test/` |
| Jumlah file gambar | 480 |
| Struktur direktori | Flat, tanpa folder kelas dan tanpa label |
| Keterbacaan gambar | 480/480 dapat dibuka |
| Format file | 479 JPEG (477 `.jpg`, 2 `.jpeg`) dan 1 PNG |
| Mode warna | 480 RGB |
| Dimensi unik | 51 |
| ID stem duplikat | Tidak ada |
| Header sample submission | `image_id`, `waste_category` |
| Baris sample submission | 480 |
| ID sample tanpa gambar | Tidak ada |
| Gambar tanpa ID sample | Tidak ada |
| ID duplikat pada sample | Tidak ada |
| Leading zero | Dipertahankan, contoh `001` sampai `480` |

Kolom `waste_category` pada sample submission memang berisi nilai. Pipeline hanya membaca header dan `image_id` untuk format dan urutan; seluruh kategori output selalu diganti oleh prediksi model.

## 3. Batas Data dan Pencegahan Leakage

| Sumber | Penggunaan |
| --- | --- |
| Train (1.340) | Training model saja |
| Validation (383) | Monitoring, early stopping, checkpoint selection, dan validation report |
| Internal test (191) | Evaluasi independen tahap akhir, bukan submission atau tuning iteratif |
| Competition test (480) | Inference Kaggle submission saja |

`CompetitionTestDataset` fail closed: hanya menerima `data/competition/test/`. Path validation, internal test, dan Roboflow ditolak. Dataset tidak membaca manifest split, anotasi Roboflow, maupun target label; setiap item hanya berupa `(image_tensor, image_id)` dalam urutan sample submission.

## 4. Arsitektur Modul

- `src/models/cnn_from_scratch.py` menyimpan `CNNFromScratch` identik dengan baseline agar training notebook dan loader checkpoint memakai arsitektur yang sama.
- `src/inference/competition.py` melakukan audit test, membaca template, memetakan ID ke file, membangun dataset unlabeled, dan membuat DataLoader sequential.
- `src/inference/predict.py` memuat checkpoint PyTorch secara read-only, memverifikasi `model_identifier` dan class mapping, mengaktifkan `eval()`, serta memprediksi dalam `torch.inference_mode()`.
- `src/inference/submission.py` memvalidasi jumlah/order ID/kategori, menulis CSV UTF-8, dan membaca ulang CSV untuk memastikan format output.

Dataset/audit dan writer submission tidak bergantung pada arsitektur CNN, sehingga model pretrained berikutnya dapat memakai preprocessing dan loader checkpoint masing-masing. YOLOv8 Classification memerlukan adapter prediksi model-specific pada TASK berikutnya, tetapi dapat memakai audit ID dan validasi/writer CSV yang sama.

## 5. CNN Checkpoint and Preprocessing

Loader CNN menggunakan `models/checkpoints/cnn_scratch_baseline_v1/best.pt`, bukan `last.pt`. Payload harus memiliki `model_identifier` `cnn_from_scratch`, delapan indeks kelas kontigu, dan class mapping yang sama dengan artefak training aktif.

Preprocessing inference CNN sama dengan validation: RGB, direct resize 224 x 224 dengan bilinear interpolation, CHW `float32`, scaling [0, 1], tanpa ImageNet normalization, dan tanpa augmentasi acak. Device memilih CUDA bila tersedia dan CPU sebagai fallback.

## 6. Manual Notebook Workflow

Bagian **Test Inference & Kaggle Submission** ditambahkan pada `notebooks/01_cnn_from_scratch.ipynb` setelah diagnostic validation. Per TASK 05D, setup inference bersifat mandiri untuk kernel baru: ia menemukan root proyek, menambahkan root ke `sys.path`, lalu membaca konfigurasi dan metadata dari artefak disk. Cell audit melakukan pemeriksaan data dan checkpoint secara read-only, tetapi tidak melakukan inference atau membuat CSV ketika flag default tetap aktif.

1. Buka notebook dari root proyek atau `notebooks/`, lalu jalankan Cell 1 Standalone Inference Setup; training tidak perlu dijalankan ulang.
2. Jalankan cell audit test resmi dan checkpoint. Perbaiki data terlebih dahulu bila audit gagal.
3. Ubah `RUN_TEST_INFERENCE = False` menjadi `True`.
4. Jalankan cell inference satu kali. Cell membangun DataLoader test, memuat `best.pt`, memprediksi 480 gambar, memvalidasi hasil, menulis CSV, menampilkan lima baris awal, distribusi kelas, dan lokasi output.
5. Set flag kembali ke `False` setelah selesai bila notebook akan dibagikan.

Flag training yang sudah ada dipertahankan: `RUN_TRAINING = False`, `RUN_RESUME = False`, dan `RUN_CONTINUED_TRAINING = False`.

## 7. Submission Validation and Storage

Target CSV CNN adalah `Final Dataset for Submission to Kaggle/submission_cnn_from_scratch.csv`. Writer menolak output jika jumlah prediksi tidak sama dengan template, ID berubah/duplikat/tertukar, kategori kosong, atau kategori di luar delapan kelas training. Writer juga menolak menimpa file target kecuali `overwrite=True` diberikan secara eksplisit; notebook menggunakan `overwrite=False`.

Setelah model lain selesai dilatih, submission langsung dapat dibuat di direktori yang sama dengan nama `submission_mobilenetv3.csv`, `submission_efficientnet.csv`, `submission_resnet50.csv`, atau `submission_yolov8.csv`. Tidak ada CSV dibuat untuk model yang belum dilatih.

## 8. Upload Manual to Kaggle

Setelah notebook menampilkan "Validasi output selesai", pengguna mengunggah `submission_cnn_from_scratch.csv` secara manual pada halaman kompetisi Kaggle. Pastikan file yang dipilih berasal dari model/eksperimen yang memang ingin dibandingkan dan jangan mengganti submission model lain.

## 9. Keterbatasan dan Risiko

- Pipeline menjamin integritas format dan pencegahan source leakage, bukan kualitas prediksi; baseline CNN tetap memiliki validation macro F1 58.69%.
- Audit tidak memberikan label atau metrik test resmi; test resmi tetap unlabeled dari sudut pipeline.
- Gambar rusak, ID tidak sinkron, struktur folder baru, atau perubahan header sample membuat audit berhenti dengan error jelas.
- `overwrite=False` mencegah overwrite tidak sengaja, sehingga pengguna perlu menghapus atau mengganti nama CSV CNN lama secara sadar sebelum membuat ulang submission CNN.

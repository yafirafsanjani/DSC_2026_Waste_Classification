# TASK 07 - EfficientNet-B2 Pretrained

## Tujuan dan batas data

Eksperimen teknis transfer learning pada dataset **V1 lama**, bukan bukti kepatuhan terhadap aturan kompetisi maksimal lima label manual per kelas. Tidak ada pseudo-labeling, split baru, V5, label test resmi, atau label test Roboflow. Manifest aktif `data/metadata/split_manifest.csv` berasal dari `data/extracted/v1/train/`: train 1.340, validation 383, dan internal test 191. Internal test hanya dihitung saat audit; tidak dipakai untuk training, tuning, evaluasi, pemilihan model, atau submission. Delapan kelas dan indeksnya ditentukan berdasarkan urutan alfabetis kelas di manifest aktif.

## Model dan preprocessing

`torchvision.models.efficientnet_b2` memakai `EfficientNet_B2_Weights.IMAGENET1K_V1`. Bobot asli disimpan di cache Torch Hub; jika belum tersedia, torchvision mengunduhnya dan kegagalan unduh menghentikan inisialisasi, tanpa fallback bobot acak. Classifier torchvision mempertahankan dropout 0,3, sedangkan linear 1408-output diganti menjadi 1408-8. Total 7.712.266 parameter, 11.272 trainable pada Phase A; setelah membuka `features.7` dan `features.8`, 3.245.524 trainable. Pemanggilan kembali checkpoint menggunakan `weights=None` lalu memuat state dictionary.

Validation dan test menggunakan `EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()`: RGB, resize sisi pendek 288, center crop 288x288, bicubic, float32 [0,1] sebelum normalisasi ImageNet mean `[0.485, 0.456, 0.406]` dan std `[0.229, 0.224, 0.225]`. Train memakai resize 288x288, flip horizontal, rotasi sampai 15 derajat, shear sampai 10 derajat, brightness 10%, dan satu normalisasi. Preview train wajib ditinjau karena warp, crop, atau border dapat mengubah ciri objek. Validation/test tidak menerima augmentasi. Perbedaan aspect ratio train vs validation merupakan keterbatasan yang perlu dipantau.

## Pelatihan manual dan monitoring

Seed 42, batch awal 8, Windows `num_workers=0`, CUDA AMP jika tersedia, gradient accumulation opsional. Tidak ada training saat notebook dibuka. Jika OOM, hentikan, ubah batch ke 4 atau 2 dan jalankan ulang secara eksplisit; resume harus diaktifkan sendiri dan hanya batch size, accumulation, dan AMP dapat berubah. State optimizer, scheduler, scaler, early stopping, RNG, dan history di-restore dari `last.pt`. DataLoader shuffle generator tidak diserialisasi terpisah, sehingga resume bukan jaminan urutan batch identik dengan run tanpa interupsi. Jangan resume run yang sudah early-stop tanpa keputusan eksperimen baru. Pengguna bertanggung jawab meninjau hasil sebelum melanjutkan.

- **Phase A**: semua backbone frozen, classifier trainable; AdamW LR 0,001 dan weight decay 0,0001; maksimal 15 epoch; early stopping minimum validation loss, patience 5. BatchNorm backbone selalu `eval`, termasuk setelah `model.train()`.
- **Phase B**: mulai dari `best.pt` Phase A, bukan last; hanya final MBConv stage `features.7` dan convolution `features.8` dibuka; AdamW baru backbone LR 0,00001, classifier LR 0,0001, weight decay 0,0001; ReduceLROnPlateau; maksimal 45 epoch, patience 10; gradient clipping 1,0. BatchNorm frozen tetap `eval`, sedangkan BatchNorm pada blok trainable dapat memperbarui statistiknya. Batch kecil 4 GB VRAM dapat membuat statistik ini berisik.

Kedua fase memakai `TrainingManager` dan monitoring universal TASK 06A: satu `tqdm.auto` bar per epoch, train loss/accuracy dan LR, kemudian ringkasan loss/accuracy, macro precision/recall/F1, LR/perubahan, waktu, best epoch/loss, counter dan status checkpoint. History persist dalam JSON, CSV dan JSONL; kurva memakai history aktual dari disk. `best.pt` berdasarkan validation loss minimum, `last.pt` menyimpan state resume. Run baru tidak menimpa artefak fase yang sudah ada. Tidak ada jaminan recovery identik jika proses mati saat penulisan beberapa artefak berbeda; periksa history/last sebelum resume.

## Checkpoint dan hasil

- `models/checkpoints/efficientnet_b2_head_v1/{best,last}.pt`
- `models/checkpoints/efficientnet_b2_finetune_v1/{best,last}.pt`
- `models/checkpoints/efficientnet_b2_selected_v1.json`: metadata pointer, tidak menyalin bobot. Pemilihan dihitung ulang dari best Phase A/B berdasarkan minimum validation loss, sehingga metadata tidak dapat membuat pilihan lama terus dipakai setelah Phase B. Hasil seri memilih Phase A.
- `outputs/metrics/efficientnet_b2_head_v1.{json,csv}` dan `outputs/metrics/efficientnet_b2_finetune_v1.{json,csv}`
- `outputs/logs/efficientnet_b2_head_v1.jsonl` dan `outputs/logs/efficientnet_b2_finetune_v1.jsonl`
- `outputs/figures/efficientnet_b2_head_v1_learning_curves.png` dan `outputs/figures/efficientnet_b2_finetune_v1_learning_curves.png`
- `outputs/metrics/efficientnet_b2_validation.json` dan `outputs/figures/efficientnet_b2_confusion_matrix.png`
- `Final Dataset for Submission to Kaggle/submission_efficientnet_b2.csv`

Evaluasi dari best terpilih melaporkan loss, accuracy, macro precision/recall/F1, classification report delapan kelas, serta confusion matrix mentah dan dinormalisasi. Heuristik overfitting bukan bukti kausal. Selection dan metrik memakai validation yang sama, sehingga hasil cenderung optimistis; dataset kecil dan label lama membatasi generalisasi. Model pretrained tetap memerlukan evaluasi independen sesuai aturan kompetisi baru. Checkpoint PyTorch dimuat dengan pickle untuk memulihkan state lengkap; hanya gunakan file checkpoint lokal yang tepercaya.

## Menjalankan notebook

1. Pastikan Python 3.11, environment `venv_gelarrasa`, PyTorch/torchvision kompatibel, Jupyter dan GPU tersedia. Jalankan `notebooks/04_efficientnet.ipynb` section 01-10 secara berurutan; section 08 memuat pretrained weights dan dapat mengunduhnya.
2. Di section 02 ubah `RUN_HEAD_TRAINING=True`, jalankan ulang section 02 dan 10-12. Section 11 melakukan training; lihat history dan kurva. Kembalikan flag `False` sebelum menjalankan ulang notebook.
3. Opsional, setelah Phase A, ubah `RUN_FINE_TUNING=True`, jalankan ulang section 02 dan 13-15. Phase B memuat best Phase A; kedua eksperimen terpisah. Set kembali flag ke `False` setelah selesai.
4. Jalankan section 15-16 untuk membandingkan dan memilih best. Setelah hanya Phase A pun selection dan submission dapat dilakukan. Ubah `RUN_VALIDATION_EVALUATION=True`, jalankan ulang section 02 lalu 17-20 untuk evaluasi dan laporan; section 07 loader validation harus tersedia.
5. Jalankan section 21-22. Di section 23 ubah `RUN_TEST_INFERENCE=True` dan jalankan manual untuk menghasilkan CSV model. Ubah kembali ke `False` sesudahnya. Section 24 memverifikasi CSV tanpa inferensi ulang. Pengguna mengunggah CSV sendiri. Nama khusus B2 ini mengikuti permintaan TASK 07 dan berbeda dari nama EfficientNet generik pada standar lama.

## Standalone inference setelah kernel baru

Jalankan hanya section 22 dari root atau folder `notebooks`; cell ini menambahkan root ke `sys.path`, mengimpor dependensi, membaca config dan mapping, memilih best Phase A/B, memuat model dan checkpoint, memeriksa preprocessing, memilih device, memuat ID sample, dan memverifikasi 480 gambar resmi. Jika belum ada best checkpoint, cell berhenti dengan pesan jelas. Lalu aktifkan dan jalankan section 23; section 24 memvalidasi tanpa prediksi ulang. Sample hanya sumber header dan urutan ID, bukan sumber label. Output menolak overwrite secara default. Verifikasi ulang CSV tidak dapat membuktikan asal prediksi secara kriptografis; provenance berasal dari pipeline inference notebook dan best checkpoint yang dipilih.

## Validasi implementasi

Jalankan `python -m pytest tests/test_efficientnet_b2.py tests/test_training_monitoring.py tests/test_training.py tests/test_callbacks.py tests/test_competition_inference.py tests/test_preprocessing.py -q` di environment yang sehat. Tes memakai tensor/gambar sintetis atau audit read-only; tidak menjalankan training dataset, internal-test evaluation, inference penuh, atau membuat submission final. `venv_gelarrasa` harus menunjuk ke instalasi Python yang tersedia sebelum dapat dipakai.

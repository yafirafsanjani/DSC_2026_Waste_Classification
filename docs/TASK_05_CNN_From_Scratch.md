# TASK 05 - CNN From Scratch Experiment Notebook

## 1. Tujuan Eksperimen

TASK 05 menyediakan satu notebook eksperimen manual pada `notebooks/01_cnn_from_scratch.ipynb`. Notebook menyiapkan baseline CNN tanpa pretrained weights untuk klasifikasi delapan kelas sampah. Tujuan implementasi adalah memberi pengguna alur eksperimen yang dapat direproduksi tanpa mengubah dataset, manifest, modul bersama, atau konfigurasi global.

Tidak ada training, validation evaluation, internal-test evaluation, inference, resume, atau continued training yang dijalankan oleh implementasi TASK ini. Semua operasi tersebut dilindungi flag opt-in default `False` dan hanya boleh dijalankan pengguna melalui notebook.

## 2. Data dan Pipeline

Notebook menggunakan manifest aktif hasil TASK 02E melalui API `create_dataloader` TASK 04. Validasi awal memeriksa 1.914 record dengan split train 1.340, validation 383, dan internal test 191; delapan kelas berasal dari metadata aktual. Sumber setiap record harus diawali `data/extracted/v1/train/`. Referensi V1/test, V5, dan path terlarang ditolak oleh assertion notebook maupun validasi DataLoader.

Train loader menggunakan preprocessing dan augmentasi on-the-fly profil `cnn` dari TASK 03. Validation loader memakai transform deterministik tanpa augmentasi acak. Internal test tidak dibuat loader dalam notebook agar tidak digunakan untuk pemilihan checkpoint atau hyperparameter.

## 3. Desain CNN

Model `CNNFromScratch` memiliki empat convolutional block dengan dua convolution 3x3 pada tiap block, Batch Normalization, ReLU, dan Max Pooling. Jumlah kanal bertumbuh dari 32 menjadi 64, 128, dan 192. Fitur diakhiri `AdaptiveAvgPool2d((1, 1))`, lalu head `Linear(192, 256)`, ReLU, Dropout 0,40, dan linear classifier.

Adaptive pooling menahan ukuran classifier agar tidak membesar pada input 224x224. Arsitektur ini cukup kecil untuk baseline pada GPU 4 GB VRAM dan memberi regularisasi awal melalui Batch Normalization serta Dropout. Jumlah output diambil dari `len(class_mapping)`, bukan angka kelas yang ditulis permanen.

Pemeriksaan sintetis menghasilkan output `float32` dengan bentuk `(2, 8)`, seluruh nilainya finite. Jumlah parameter trainable adalah **892.648**. Nilai tersebut adalah hasil pemeriksaan model kosong dengan tensor sintetis, bukan hasil eksperimen kompetisi.

## 4. Konfigurasi Awal

- Image size: 224x224.
- Batch size: 16; dokumentasi notebook memberi opsi 8 atau 4 untuk CUDA OOM.
- Maximum epochs: 100.
- Optimizer: AdamW, learning rate 0,001, weight decay 0,0001.
- Loss: CrossEntropyLoss.
- Random seed: 42.
- AMP: aktif bila CUDA tersedia.
- Gradient accumulation: 1 dan clipping norm: 1,0 sebagai baseline yang dapat diubah di notebook.
- Early stopping: `val_loss`, mode `min`, patience 10, min delta 0,001.
- Scheduler: ReduceLROnPlateau, factor 0,5, patience 4, minimum learning rate `1e-6`.

Nilai ini tinggal di dictionary notebook dan tidak mengubah YAML global. `EXPERIMENT_ID` harus diganti untuk eksperimen baru supaya metrics, logs, dan checkpoint tidak tertimpa.

## 5. Integrasi TASK 03 dan TASK 04

Notebook memakai `resolve_model_config` dan `load_active_class_mapping` dari preprocessing TASK 03, serta `create_dataloader` dari TASK 04. Training tidak membuat loop baru: cell manual memanggil `TrainingManager.fit`, yang mengoordinasikan `train_one_epoch`, `validate_one_epoch`, scheduler, early stopping, checkpoint best/last, metrics history, dan log per epoch.

Best checkpoint disimpan menurut `val_loss`; notebook tidak mengasumsikan best accuracy. History aktual dapat dibaca dengan `load_history` dan divisualisasikan dengan `plot_learning_curves`. Cell evaluasi validation memuat best checkpoint lalu memanggil `validate_one_epoch` untuk loss, accuracy, macro precision, macro recall, macro F1, classification report, dan confusion matrix dalam urutan class mapping yang sama.

## 6. Resume dan Continued Training

Cell resume bersifat opsional dan default `RUN_RESUME = False`. Bila pengguna mengaktifkannya, `resume_from_checkpoint` memulihkan model, optimizer, scheduler, AMP GradScaler, epoch, early stopping, history, konfigurasi, class mapping, dan state RNG dari `last.pt`. Notebook hanya menampilkan epoch lanjutan; pengguna tetap harus memanggil `manager.fit` secara eksplisit untuk benar-benar melanjutkan eksperimen.

Untuk CNN from scratch, cell lanjutan memakai istilah continued training, bukan fine-tuning pretrained backbone. Ia membuat experiment ID baru, memuat `best.pt` melalui pemeriksaan kompatibilitas, membuat optimizer/scheduler/early stopping baru, dan menggunakan learning rate yang lebih kecil. Fase ini juga default tidak aktif.

## 7. Monitoring dan Visualisasi

Saat training dijalankan pengguna, `TrainingManager` menyimpan history aktual pada `outputs/metrics/{experiment_id}.json` dan log JSONL pada `outputs/logs/{experiment_id}.jsonl`. History mencakup loss/accuracy train, loss/accuracy/macro-F1 validation, learning rate, durasi, best epoch, status checkpoint, dan status early stopping.

Notebook hanya menghasilkan learning curve apabila history nyata ditemukan. Bila belum ada training, cell menampilkan pesan informatif. Tidak ada angka akurasi, loss, F1, confusion matrix, atau learning curve fiktif pada notebook maupun dokumentasi ini.

## 8. Cara Menjalankan

1. Aktifkan `venv_gelarrasa`, lalu buka `notebooks/01_cnn_from_scratch.ipynb` melalui Antigravity IDE.
2. Jalankan cell dari awal sampai konfigurasi, validasi data, loader, preview, dan inisialisasi manager.
3. Tinjau batch size serta penggunaan GPU; ubah hanya konfigurasi notebook bila diperlukan.
4. Set `RUN_TRAINING = True` dan jalankan cell training secara manual.
5. Setelah training, baca history dan learning curve; gunakan validation evaluation manual bila best checkpoint tersedia.
6. Jangan gunakan internal test untuk tuning. Resume atau continued training memerlukan keputusan eksplisit pengguna.

## 9. Pengujian Statis Aktual

Notebook tervalidasi sebagai Jupyter Notebook v4 dengan **42 cell**: 23 Markdown dan 19 code cell. Seluruh code cell berhasil dikompilasi tanpa dieksekusi. Notebook memiliki 22 heading bagian yang mengikuti urutan eksperimen. Notebook tidak menyimpan output eksekusi. Flag `RUN_TRAINING`, `RUN_VALIDATION_EVALUATION`, `RUN_RESUME`, dan `RUN_CONTINUED_TRAINING` semuanya berstatus `False` secara default.

Arsitektur diuji terpisah dengan tensor sintetis CPU, bukan DataLoader kompetisi: input `(2, 3, 224, 224)` menghasilkan `(2, 8)` dengan dtype `torch.float32` dan nilai finite. Pengujian tidak menjalankan backward pass atau optimizer step.

## 10. Keterbatasan dan Langkah Berikutnya

Notebook belum memiliki hasil performa karena training memang belum dijalankan. Kualitas arsitektur, learning rate, augmentasi, dan regularisasi hanya dapat dinilai setelah eksperimen nyata yang dikendalikan pengguna. Setelah baseline selesai, langkah berikutnya adalah merekam hasil validation secara disiplin dan membandingkan eksperimen tanpa memakai internal test berulang kali.

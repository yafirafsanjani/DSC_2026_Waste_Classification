# TASK 04 - Training Utilities dan Callbacks

## Identitas dan Tujuan

TASK 04 menyiapkan utilitas pelatihan PyTorch yang modular untuk eksperimen notebook pengguna. Implementasi ini tidak menjalankan pelatihan pada dataset kompetisi. Tujuannya adalah memastikan eksperimen memakai dataset aktif yang konsisten, transform yang tepat untuk model, metrik yang dapat ditelusuri, dan checkpoint yang dapat dipulihkan.

## Kondisi Dataset dan Batasan

Dataset aktif tetap hasil TASK 02E: 1.914 salinan terverifikasi dari `data/extracted/v1/train`, yaitu 1.340 train, 383 validation, dan 191 internal test dengan delapan kelas. `data/extracted/v1/test`, seluruh V5, serta dua gambar label-noise ND-002 tidak termasuk manifest aktif. TASK ini tidak menulis ke `data/processed`, tidak mengubah manifest, tidak membuka notebook, dan tidak menjalankan training atau evaluasi internal test pada data kompetisi.

## Arsitektur

- `src/data/dataloader.py` menyediakan `ManifestImageDataset`, `load_manifest_records`, dan `create_dataloader`.
- `src/training/train.py` berisi loop satu epoch untuk train; `src/training/validate.py` berisi validasi tanpa gradien.
- `src/callbacks/training_manager.py` menjadi koordinator eksperimen yang memanggil loop tersebut, tanpa menduplikasi logika.
- `src/callbacks/early_stopping.py`, `model_checkpoint.py`, dan `lr_scheduler.py` memisahkan state penghentian, persistensi checkpoint, dan scheduler.
- `src/training/resume.py` serta `fine_tuning.py` hanya menyiapkan pemulihan/persiapan fase eksplisit; keduanya tidak memulai training otomatis.
- `src/evaluation/metrics.py` menghitung metrik seluruh prediksi; `visualization.py` hanya memplot history nyata.

## DataLoader dan TASK 03

`ManifestImageDataset` membaca `data/metadata/split_manifest.csv`, menolak sumber selain `data/extracted/v1/train`, memeriksa layout `data/processed/{split}/{class}/`, memvalidasi file, dan mengambil class mapping alfabetis yang sama di seluruh split. Default `num_workers=0` digunakan sebagai pilihan aman pada Windows. Generator, seed worker, train shuffle, dan validation/internal-test tanpa shuffle membuat loader dapat direproduksi.

Loader memakai `build_transforms` dan `build_split_augmentation` dari TASK 03. Augmentasi hanya aktif pada train; transform validation atau internal test yang membawa augmentasi acak ditolak. Konfigurasi model tetap berasal dari YAML sehingga profil CNN, MobileNetV3, EfficientNet, dan ResNet berbeda tanpa mengubah kode bersama. YOLOv8 tetap memakai adapter Ultralytics TASK 03 dan tidak dipaksa ke custom loop ini.

## Loop Pelatihan dan Validasi

`train_one_epoch` menyediakan forward pass, loss, backward pass, optimizer step, `tqdm`, AMP CUDA opsional, gradient clipping, dan gradient accumulation. Loss train dijumlahkan dengan bobot jumlah sampel. Window accumulation terakhir dihitung dari jumlah batch aktual sehingga skala gradient tidak salah saat jumlah batch tidak habis dibagi faktor accumulation. Loss non-finite menghentikan epoch dengan error yang jelas.

`validate_one_epoch` memanggil `model.eval()` dan `torch.inference_mode()`. Tidak ada backward pass atau optimizer step. Loss validasi ditimbang berdasarkan jumlah sampel; accuracy, macro precision, macro recall, macro F1, metrik per kelas, confusion matrix, dan classification report dihitung dari seluruh target/prediksi, bukan rata-rata batch. Internal test tidak dipanggil manager dan tidak boleh dipakai untuk memilih checkpoint.

## Early Stopping, Checkpoint, dan Resume

`EarlyStopping` default memonitor `val_loss` dengan mode `min`, patience 10, serta `min_delta` 0,001. State menyimpan best metric, best epoch, jumlah epoch tanpa perbaikan, dan alasan penghentian sehingga bisa dipulihkan. Berhentinya early stopping tidak memulai kembali epoch atau fase baru secara otomatis.

`ModelCheckpoint` menyimpan `best.pt` dan `last.pt` pada `models/checkpoints/{experiment_id}/`. Penulisan memakai berkas sementara lalu `Path.replace()` agar target tidak tertimpa checkpoint parsial. Payload memuat state model, optimizer, scheduler, AMP scaler, early stopping, epoch, best metric, konfigurasi, model identifier, class mapping, history, seed, dan state RNG. Checkpoint terbaik dipilih oleh monitor, bukan oleh accuracy secara implisit.

`resume_from_checkpoint` memeriksa model identifier, class mapping, dan konfigurasi apabila diberikan; kemudian memulihkan model, optimizer, scheduler, scaler, callback, dan RNG. Nilai kembali `start_epoch` adalah epoch terakhir ditambah satu. Pengguna menentukan sendiri apakah dan kapan resume dilakukan.

## Scheduler dan Fine-Tuning

`configs/callbacks.yaml` menyediakan baseline ReduceLROnPlateau: monitor `val_loss`, factor 0,5, patience 4, dan minimum learning rate `1e-6`. Integrasi juga mendukung `CosineAnnealingLR`; plateau diberi metrik pada akhir epoch sedangkan cosine di-step sekali tiap epoch. Learning rate aktual seluruh parameter group disimpan dalam history.

Utilitas fine-tuning menyediakan pembekuan backbone, pembukaan seluruh atau child terakhir backbone, parameter group backbone/classifier dengan learning rate terpisah, dan pemuatan best checkpoint yang kompatibel. Struktur arsitektur tidak diasumsikan; notebook perlu secara eksplisit memberikan modul backbone dan classifier yang benar. Fase baru memakai optimizer, scheduler, dan early stopping baru secara sengaja.

## Monitoring dan Visualisasi

Saat `TrainingManager.fit` dipanggil dari notebook, setiap epoch mencatat epoch, train loss/accuracy, validation loss/accuracy, validation macro precision/recall/F1, learning rate, durasi, best epoch, status checkpoint, dan status early stopping. History aktual disimpan sebagai `outputs/metrics/{experiment_id}.json`; satu record per baris disimpan sebagai `outputs/logs/{experiment_id}.jsonl`. Kedua berkas tidak dibuat ketika modul hanya diimpor atau ketika belum ada training.

`plot_learning_curves` menolak history kosong dan tidak membuat grafik sintetis. Bila history nyata tersedia, fungsi menghasilkan grafik train-vs-validation loss, train-vs-validation accuracy, learning rate, serta validation macro F1, disertai penanda best validation loss dan early stopping jika terjadi.

## Preview Augmentasi

`create_augmentation_preview_all_classes` menambah preview representatif untuk delapan kelas. Fungsi memilih satu record dari split `train` pada manifest aktif, lalu membuat original, preprocessing CNN 224x224, dan contoh augmentasi training secara on-the-fly. Artefak tersimpan pada `outputs/figures/augmentation_preview_all_classes.png`. Preview menggunakan gambar aktif yang nyata tetapi tidak menulis kembali gambar ke dataset.

## Kontrak API untuk Notebook

```python
from src.data.dataloader import create_dataloader
from src.callbacks.training_manager import TrainingManager

train_loader = create_dataloader("train", "cnn", batch_size=32, seed=42)
val_loader = create_dataloader("val", "cnn", batch_size=32, seed=42)
manager = TrainingManager(model, optimizer, criterion, class_mapping, "cnn", "cnn_baseline_001")
history = manager.fit(train_loader, val_loader, max_epochs=50)
```

Panggilan ini adalah kontrak untuk notebook pengguna, bukan perintah yang dijalankan otomatis oleh kode proyek. Untuk YOLOv8 Classification, gunakan trainer, callback, `best.pt`, `last.pt`, `resume`, dan `results.csv` milik Ultralytics. Adapter TASK 03 hanya meneruskan parameter yang sesuai dan mencegah augmentasi eksternal kedua.

## Pengujian Aktual

Pemeriksaan sintaks dijalankan untuk modul TASK 04. Suite sintetis `tests/test_dataloader.py`, `tests/test_callbacks.py`, dan `tests/test_training.py` menghasilkan **10 passed**. Pengujian mencakup class mapping, sumber manifest terlarang, shuffle/determinisme validation, early stopping, checkpoint best/last, resume kompatibel/tidak kompatibel, scheduler, gradient accumulation, metrik/confusion matrix, serialisasi history, plot dari history nyata, dan loss non-finite. Pengujian memakai direktori sementara serta tensor/gambar sintetis; tidak melatih model pada gambar kompetisi.

## Keterbatasan dan Rekomendasi TASK 05

Implementasi tidak dapat membuktikan kualitas model tanpa eksperimen yang dijalankan pengguna. Integrasi metadata pretrained serta kebijakan augmentasi model-specific tetap mengikuti TASK 03 dan belum divalidasi melalui training. Pada TASK 05, notebook dapat menambahkan arsitektur dan konfigurasi eksperimen per model, memakai validation untuk pemilihan model, lalu menggunakan internal test hanya untuk evaluasi final yang disetujui pengguna.

# TASK 01 - Dataset Audit

## 1. Identitas dan Tujuan Task

TASK 01 membangun audit integritas data untuk dua ekspor Roboflow yang tersedia saat itu, V1 dan V5. Tujuannya adalah mendokumentasikan struktur aktual, distribusi kelas, integritas berkas, exact duplicate, near-duplicate, dan indikasi leakage sebelum ada keputusan cleaning atau training.

## 2. Latar Belakang Masalah

Ekspor V5 diketahui merupakan hasil resize dan augmentasi dari data Roboflow. File yang berbeda tidak cukup untuk membuktikan bahwa gambar independen. Selain itu, V1/test dan V5/test telah berlabel sehingga tidak boleh menjadi data pembelajaran maupun pemilihan model.

## 3. Dataset atau Input yang Digunakan

Audit membaca `data/extracted/v1/` dan `data/extracted/v5/` secara read-only. Laporan audit tersimpan di `data/metadata/dataset_audit_summary.json`, `dataset_audit_image_manifest.csv`, serta laporan duplicate dan leakage terkait.

## 4. Metodologi dan Tahapan Pengerjaan

1. Memindai folder dataset, split, dan folder kelas aktual.
2. Membuka setiap berkas gambar dengan Pillow untuk mendeteksi gambar tidak terbaca dan ketidaksesuaian ekstensi/format.
3. Menghitung SHA-256 untuk exact duplicate.
4. Menghitung DCT perceptual hash dan Hamming distance untuk kandidat near-duplicate.
5. Mengekstrak identitas sumber dari pola nama Roboflow sebelum `.rf.<id>`.
6. Membandingkan konteks dataset/split untuk menghasilkan evidence leakage tanpa menggunakan label test untuk keputusan model.

## 5. Penjelasan File yang Dibuat atau Dimodifikasi

- `src/data/data_audit.py`: auditor modular, CLI, hashing, laporan CSV/JSON, dan grafik distribusi.
- `tests/test_dataset.py`: pada tahap awal memverifikasi parsing identitas Roboflow, integritas, duplicate, leakage, dan output laporan dengan fixture sintetis.
- `data/metadata/dataset_audit_*.csv` dan `dataset_audit_summary.json`: manifest dan ringkasan audit.
- `outputs/figures/dataset_class_distribution.png`: grafik distribusi kelas/split.

## 6. Parameter dan Konfigurasi Penting

- Dataset yang diaudit: V1 dan V5.
- Ambang perceptual hash: Hamming distance maksimum 8.
- SHA-256 digunakan sebagai bukti byte-identik.
- Test split hanya dibaca untuk audit leakage; tidak dipakai untuk training atau evaluasi.

## 7. Hasil Implementasi dan Artefak

Audit menemukan 8.146 gambar yang dapat dibaca: V1 terdiri dari train 1.916 dan test 480; V5 terdiri dari train 5.031, valid 479, dan test 240. Delapan kelas yang sama ditemukan pada kedua versi: cardboard, electronic, foodwaste, glass, metal, paper, plastic, dan textile.

Tidak ada gambar rusak atau ekstensi tidak valid dalam folder kelas. Ditemukan 2.396 kelompok exact duplicate dan 2.396 kelompok identitas sumber Roboflow lintas konteks. Pemetaan exact/source penting adalah 1.677 kelompok V1/train-V5/train, 239 kelompok V1/train-V5/valid, 240 kelompok V1/test-V5/valid, dan 240 kelompok V1/test-V5/test. Audit juga mencatat 1.333 pasangan near-duplicate, dengan 356 evidence perceptual-hash lintas konteks.

## 8. Hasil Pengujian

Auditor memiliki fixture test sintetis dalam `tests/test_dataset.py`. Suite regresi proyek kemudian diperluas pada task berikutnya; rerun final setelah TASK 02C menghasilkan `6 passed`.

## 9. Temuan, Masalah, serta Keterbatasan

V5 tidak independen dari V1. Khususnya, V5/valid mengandung gambar yang berasal dari V1/test dan V1/train. Ini membatalkan penggunaan V5 sebagai validation set untuk eksperimen berbasis V1. Perceptual hash adalah kandidat visual, bukan bukti identitas sumber tanpa verifikasi tambahan.

## 10. Keputusan Teknis dan Alasannya

V1/train dipilih sebagai satu-satunya kandidat sumber untuk tahap berikutnya. V1/test dan seluruh V5 dikarantina dari pipeline pembelajaran. Keputusan ini memprioritaskan independensi evaluasi atas jumlah data yang lebih besar tetapi terkontaminasi.

## 11. Kesimpulan dan Hubungan dengan Task Berikutnya

TASK 01 menyediakan bukti bahwa V1/train harus dibersihkan dan di-split secara mandiri. TASK 02 kemudian menggunakan hanya 1.916 gambar V1/train untuk cleaning dan split awal.

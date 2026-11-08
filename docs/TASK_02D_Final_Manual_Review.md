# TASK 02D - Final Manual Review

## 1. Identitas dan Tujuan Task

TASK 02D menyiapkan bahan keputusan manual untuk tiga kandidat near-duplicate ambigu lintas split (`ND-021`, `ND-023`, dan `ND-025`) serta satu kandidat potensi label noise (`ND-002`). Task ini tidak menentukan hubungan sumber gambar atau ketepatan label secara otomatis.

## 2. Latar Belakang Masalah

TASK 02C telah mengisolasi seluruh hubungan visual yang dikonfirmasi kuat. Tiga kandidat `metal` masih berada pada train dan validation karena bukti perceptual hash saja tidak cukup untuk membuktikan bahwa gambar berasal dari sumber yang sama. Selain itu, `ND-002` memiliki pHash identik tetapi label `cardboard` dan `paper`; pasangan ini sudah berada dalam internal test yang sama, sehingga isu utamanya adalah kemungkinan label noise, bukan leakage lintas split.

## 3. Dataset atau Input yang Digunakan

Input dibatasi pada `data/metadata/near_duplicate_split_review.csv`, `data/metadata/near_duplicate_groups.csv`, `data/metadata/split_manifest.csv`, dan gambar sumber yang jalurnya diawali `data/extracted/v1/train/`. Generator memvalidasi mapping setiap gambar terhadap split manifest sebelum membuat artefak. Dataset `v1/test` dan seluruh `v5` tidak dibaca.

## 4. Metodologi dan Tahapan Pengerjaan

1. Memilih empat ID pasangan dalam urutan review manual yang ditentukan.
2. Memvalidasi split, kelas, SHA-256, source identity, dan perceptual hash pasangan terhadap manifest aktif.
3. Memeriksa ulang jarak pHash dari kedua nilai hash yang tercatat.
4. Merender dua gambar sumber per pasangan ke contact sheet resolusi tinggi tanpa menulis kembali gambar sumber.
5. Membuat CSV review dengan kolom keputusan, alasan, reviewer, dan waktu keputusan yang sengaja dikosongkan.
6. Tidak melakukan cleaning, relabeling, pemindahan gambar, atau resplitting.

## 5. Penjelasan File yang Dibuat atau Dimodifikasi

- `src/data/final_manual_review.py` menyediakan generator deterministik, pemeriksaan provenance V1/train, verifikasi metadata manifest, CSV template, dan contact sheet.
- `data/metadata/final_manual_review.csv` menyimpan empat baris bukti dan kolom keputusan pengguna.
- `outputs/figures/final_manual_review.png` menampilkan bukti visual berpasangan beserta ID, kelas, split, nama file, source identity, dan prefix SHA-256.
- `docs/README.md` diperbarui agar TASK 02D tercantum pada indeks dan status data mencerminkan bahwa keputusan manual masih tertunda.

## 6. Parameter dan Konfigurasi Penting

Generator menetapkan urutan `ND-021`, `ND-023`, `ND-025`, dan `ND-002`. Contact sheet dibuat sebagai satu PNG berukuran 3.600 x 6.230 piksel. Gambar hanya dibaca dari jalur sumber V1/train dan diperbesar untuk keterbacaan tampilan; tidak ada perubahan pada file sumber, split manifest, atau label.

## 7. Hasil Implementasi dan Artefak

- `ND-021`: `metal` train (`0944_jpg.rf.e05e26cf73a7d3a03484f08d670fece1.jpg`) versus `metal` validation (`1489_jpg.rf.ea9e0fb00b0f3c6e34c2feac588d04a0.jpg`), jarak pHash 4.
- `ND-023`: `metal` train (`0991_jpg.rf.3c2f728d2f2b111a524e800e16bedd7b.jpg`) versus `metal` validation (`1489_jpg.rf.ea9e0fb00b0f3c6e34c2feac588d04a0.jpg`), jarak pHash 4.
- `ND-025`: `metal` train (`1262_jpg.rf.1ab88a9dc72bd80eec4f1a88ada503f7.jpg`) versus `metal` validation (`1489_jpg.rf.ea9e0fb00b0f3c6e34c2feac588d04a0.jpg`), jarak pHash 2.
- `ND-002`: `cardboard` internal test (`1368_jpg.rf.1ca9d9e39e2ab975d0371ac6387ada31.jpg`) versus `paper` internal test (`1548_jpg.rf.097efc120796cf28c589550dab63ec5a.jpg`), jarak pHash 0. Pasangan ini tercatat pada kelompok visual `visual:ddbc689c479128c4` dan sudah terisolasi dalam satu split.

Seluruh pasangan mempunyai SHA-256 dan source identity yang berbeda. Contact sheet dan CSV final tersedia pada jalur yang ditentukan tanpa mengubah artefak split.

## 8. Hasil Pengujian

Modul berhasil melewati pemeriksaan sintaks Python dan dieksekusi dari root proyek. Output berisi tepat empat baris dengan urutan target yang benar, seluruh path sumber diawali `data/extracted/v1/train/`, seluruh nilai `user_decision` kosong, dan contact sheet berhasil dibuat pada ukuran 3.600 x 6.230 piksel.

## 9. Temuan, Masalah, serta Keterbatasan

Visual `ND-021`, `ND-023`, dan `ND-025` menunjukkan komposisi objek kaleng yang mirip, tetapi perbedaan varian warna, watermark, atau aset gambar tidak dapat diputuskan hanya dari pHash. Karena itu, metrik jarak 4, 4, dan 2 diperlakukan sebagai petunjuk review, bukan bukti source identity. `ND-002` memperlihatkan bukti visual sangat kuat untuk objek yang sama, tetapi penentuan label yang benar membutuhkan kebijakan label lomba atau peninjauan anotasi asli. CSV tidak mengubah atau mengisi keputusan tersebut secara otomatis.

## 10. Keputusan Teknis dan Alasannya

Contact sheet menggunakan gambar sumber V1/train agar keputusan dapat ditelusuri kembali ke input yang diizinkan. Validasi manifest dijadikan gerbang sebelum rendering untuk mencegah review atas file di luar split final. Kolom keputusan dipisahkan dari bukti teknis sehingga keputusan manusia dapat terdokumentasi tanpa mengubah hash, label, atau split yang telah diverifikasi.

## 11. Kesimpulan dan Hubungan dengan Task Berikutnya

TASK 02D selesai pada tahap persiapan keputusan manual. Dataset belum boleh disebut sepenuhnya bebas leakage secara absolut sampai pengguna mengisi keputusan untuk `ND-021`, `ND-023`, dan `ND-025`. `ND-002` juga memerlukan keputusan tentang ketepatan label, tetapi tidak memerlukan resplitting karena kedua gambar sudah berada di internal test yang sama. Tidak ada tindakan lanjutan otomatis yang dilakukan.

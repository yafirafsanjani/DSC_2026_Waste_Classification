# TASK 02E - Final Dataset Resolution

## 1. Identitas dan Tujuan Task

TASK 02E menerapkan keputusan visual pengguna dari TASK 02D untuk menyelesaikan tiga kandidat leakage ambigu dan satu kasus label noise. Tujuannya adalah menghasilkan dataset aktif yang hanya menggunakan gambar V1/train, tidak memisahkan hubungan visual yang telah dikonfirmasi, serta tidak menggunakan dua record berlabel konflik sampai verifikasi label tersedia.

## 2. Latar Belakang Masalah

TASK 02C telah mengisolasi 35 pasangan visual terkonfirmasi, tetapi `ND-021`, `ND-023`, dan `ND-025` tetap ambigu lintas train-validation. TASK 02D menyediakan contact sheet dan template keputusan. Pengguna kemudian menetapkan ketiganya sebagai `same_source_or_derivative`. Pengguna juga menetapkan `ND-002` sebagai `label_noise_confirmed` karena objek yang tampak identik memiliki label `cardboard` dan `paper`.

## 3. Dataset atau Input yang Digunakan

Resolver membaca metadata TASK 02D, cleaning manifest melalui proses cleaning V1/train, near-duplicate review, near-duplicate groups, dan split manifest aktif. Satu-satunya sumber gambar adalah `data/extracted/v1/train/`. Dataset `data/extracted/v1/test/` dan seluruh `data/extracted/v5/` tidak dibaca atau dipakai untuk split aktif.

## 4. Metodologi dan Tahapan Pengerjaan

1. Menulis keputusan pengguna ke `final_manual_review.csv` beserta rasional dan waktu pencatatan.
2. Menambahkan `ND-021`, `ND-023`, dan `ND-025` ke registry relasi visual terkonfirmasi.
3. Mengeluarkan dua source path `ND-002` dari record aktif tanpa menghapus gambar sumber atau mengubah label sumber.
4. Membentuk connected component source identity dan relasi visual terkonfirmasi, lalu melakukan group-aware stratified splitting dengan seed 42.
5. Mengarsipkan manifest, summary, groups, review, report, dan manual-review aktif sebelum promosi split baru.
6. Menyalin split ke staging, memverifikasi SHA-256 seluruh salinan, mempromosikan staging dengan rollback, lalu memverifikasi direktori hasil promosi.
7. Menulis manifest aktif, split summary, near-duplicate groups, dan resplitting report baru.

## 5. Penjelasan File yang Dibuat atau Dimodifikasi

- `src/data/data_split.py` memperluas registry visual terkonfirmasi dengan `ND-021`, `ND-023`, dan `ND-025`.
- `src/data/final_dataset_resolution.py` merekam keputusan, mengecualikan `ND-002`, menjalankan resplitting staging dengan rollback, membuat arsip metadata, serta memvalidasi provenance dan hash sumber.
- `tests/test_dataset.py` menambahkan test untuk exclusion label noise, isolasi tiga pasangan visual yang diputuskan, determinisme seed 42, validasi copy, dan ketidakberubahan file sumber.
- `data/metadata/final_manual_review.csv` sekarang berisi keputusan pengguna.
- `data/metadata/split_manifest.csv`, `data/metadata/split_summary.json`, `data/metadata/near_duplicate_groups.csv`, dan `data/metadata/resplitting_report.json` merepresentasikan dataset aktif TASK 02E.

## 6. Parameter dan Konfigurasi Penting

Resplitting memakai seed 42 dengan target train 70%, validation 20%, dan internal test 10%. Dari 1.916 record bersih V1/train, dua record `ND-002` dikeluarkan dari dataset aktif sehingga total aktif menjadi 1.914. Tiga pasangan yang diputuskan sebagai `same_source_or_derivative` membentuk satu komponen visual `metal` beranggotakan empat gambar dan wajib berada pada satu split.

## 7. Hasil Implementasi dan Artefak

Keputusan tersimpan sebagai berikut: `ND-021`, `ND-023`, dan `ND-025` = `same_source_or_derivative`; `ND-002` = `label_noise_confirmed`. Registry sekarang memuat 38 pasangan visual terkonfirmasi, dengan 37 pasangan aktif setelah `ND-002` dikeluarkan. Dataset aktif memiliki 33 kelompok visual terkonfirmasi.

Dua record yang dikecualikan adalah `data/extracted/v1/train/cardboard/1368_jpg.rf.1ca9d9e39e2ab975d0371ac6387ada31.jpg` dan `data/extracted/v1/train/paper/1548_jpg.rf.097efc120796cf28c589550dab63ec5a.jpg`. Keduanya tetap berada pada sumber mentah dan tercatat di `resplitting_report.json` dengan status `excluded_pending_label_verification`.

Arsip pra-TASK 02E menggunakan checksum, termasuk `split_manifest_pre_task_02e_597b38b5223a.csv` dan `split_summary_pre_task_02e_f65faa014849.json`. Arsip tambahan untuk review, groups, report, dan manual review juga disimpan di `data/metadata/`.

## 8. Distribusi Aktif dan Deviasi Rasio

Jumlah aktif adalah train 1.340 (70,010%), validation 383 (20,010%), dan internal test 191 (9,979%). Target integer global dengan largest remainder adalah 1.340 / 383 / 191, sehingga deviasi jumlah gambar adalah nol.

Distribusi kelas adalah: `cardboard` 183 / 52 / 26; `electronic` 143 / 41 / 20; `foodwaste` 172 / 49 / 24; `glass` 145 / 41 / 21; `metal` 167 / 48 / 24; `paper` 156 / 45 / 22; `plastic` 222 / 64 / 32; dan `textile` 152 / 43 / 22. Urutan setiap kelas adalah train / validation / internal test. Deviasi terhadap target integer per kelas adalah nol.

## 9. Hasil Pengujian dan Verifikasi Leakage

Validasi manifest aktif mencatat 1.914 record, provenance hanya V1/train, tidak ada path sumber duplikat, tidak ada SHA-256 lintas split, dan tidak ada source atau visual group aktif yang tersebar lintas split. Hash seluruh file sumber cocok dengan metadata cleaning setelah proses berjalan. Staging serta direktori hasil promosi memuat seluruh file manifest dan setiap SHA-256 salinan cocok.

Unit test mencakup komponen visual tiga pasangan baru dalam satu split, exclusion label noise dari manifest aktif, validasi copy, provenance, hash sumber, dan alokasi deterministik. Suite lengkap setelah TASK 02E menghasilkan 7 test lulus.

## 10. Keputusan Teknis dan Keterbatasan

Hubungan `ND-021`, `ND-023`, dan `ND-025` diperlakukan sebagai komponen tunggal berdasarkan keputusan visual pengguna, bukan pHash semata. `ND-002` tidak direlabel secara otomatis; exclusion menjaga training dan evaluasi dari label konflik tanpa kehilangan jejak gambar sumber. Lima kandidat pHash ambigu yang tersisa dan pasangan false positive tetap terdokumentasi, tetapi tidak diubah otomatis karena belum ada keputusan pengguna tambahan.

## 11. Kesimpulan dan Hubungan dengan Task Berikutnya

TASK 02E menghasilkan dataset aktif yang terisolasi terhadap seluruh hubungan visual yang telah dikonfirmasi, termasuk tiga keputusan manual baru. Dua record label-noise `ND-002` tidak ada di split aktif tetapi tetap dapat ditelusuri. Dataset aktif siap untuk pemeriksaan pengguna sebelum tahap berikutnya, dengan batasan bahwa status label final untuk `ND-002` masih perlu diputuskan dan kandidat ambigu yang tersisa tetap menjadi catatan audit. Tidak ada training, preprocessing, augmentasi, atau TASK 03 yang dijalankan.

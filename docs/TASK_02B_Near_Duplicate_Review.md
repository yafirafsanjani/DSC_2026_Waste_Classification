# TASK 02B - Near-Duplicate Review

## 1. Identitas dan Tujuan Task

TASK 02B memeriksa 47 kandidat near-duplicate V1/train terhadap split hasil TASK 02. Tujuannya adalah membedakan kandidat dalam split yang sama, kandidat lintas split, serta false positive yang tidak boleh langsung dianggap leakage.

## 2. Latar Belakang Masalah

SHA-256 dan identitas Roboflow pada TASK 02 tidak menangkap semua gambar yang berasal dari foto/subjek sama tetapi telah diekspor sebagai file berbeda. Perceptual hash memberi sinyal kandidat, namun membutuhkan inspeksi visual.

## 3. Dataset atau Input yang Digunakan

Input terdiri dari `cleaning_report.csv`, `cleaning_manifest.csv`, `split_manifest.csv`, dan salinan gambar V1 pada `data/processed/`. V1/test dan V5 tidak digunakan.

## 4. Metodologi dan Tahapan Pengerjaan

1. Membaca baris `near_duplicate` dari cleaning report.
2. Memetakan kedua jalur sumber setiap pasangan ke split final melalui split manifest.
3. Menghitung ulang Hamming distance dari perceptual hash pada manifest.
4. Mengelompokkan pasangan ke enam kombinasi split.
5. Membuat contact sheet pasangan untuk inspeksi visual tanpa memodifikasi gambar.

## 5. Penjelasan File yang Dibuat atau Dimodifikasi

- `src/data/near_duplicate_review.py`: join manifest, verifikasi distance, klasifikasi kombinasi split, dan generator contact sheet.
- `data/metadata/near_duplicate_split_review.csv`: laporan per pasangan.
- `outputs/figures/near_duplicate_contact_sheet.png` sampai `_08.png`: contact sheet berhalaman.
- `tests/test_dataset.py`: test kombinasi split dan manual-review flag.

## 6. Parameter dan Konfigurasi Penting

- Ambang kandidat diwariskan dari cleaning: pHash Hamming distance <= 8.
- Contact sheet berisi enam pasangan per halaman.
- Candidate lintas split diberi flag review manual, bukan langsung dihapus.

## 7. Hasil Implementasi dan Artefak

Review awal menemukan 47 pasangan: TrainTrain 19, TrainValidation 15, TrainInternal Test 5, ValidationValidation 4, ValidationInternal Test 3, dan Internal TestInternal Test 1. Sebanyak 23 pasangan lintas split diberi manual review. Distribusi distance adalah 0 sebanyak 19, 2 sebanyak 10, 4 sebanyak 7, 6 sebanyak 5, dan 8 sebanyak 6.

Dari inspeksi visual, 19 pasangan lintas split dinilai berisiko tinggi dan empat pasangan (`ND-003`, `ND-012`, `ND-044`, `ND-046`) dinilai false positive visual. Tidak ada pasangan yang memiliki SHA-256 atau identitas Roboflow sama; kesimpulan visual tidak hanya didasarkan pada hash.

Review pra-resplit diarsipkan sebagai `data/metadata/near_duplicate_split_review_pre_task_02c_ef81ba6a1abd.csv` sebelum laporan aktif diperbarui terhadap split baru.

## 8. Hasil Pengujian

Reviewer diuji melalui fixture manifest sintetis. Suite regresi proyek pada akhir TASK 02C menghasilkan `6 passed`.

## 9. Temuan, Masalah, serta Keterbatasan

Sebagian kandidat dengan distance rendah benar-benar menunjukkan foto atau subjek yang sama. Sebagian lain, terutama objek produk dengan latar putih atau variasi warna, bersifat ambigu atau false positive. Oleh karena itu pHash tidak dipakai sebagai aturan grouping otomatis.

## 10. Keputusan Teknis dan Alasannya

Pasangan risiko tinggi dijadikan input eksplisit untuk union-find pada TASK 02C. False positive tidak digabung. Kandidat ambigu dipertahankan sebagai daftar review agar pipeline tidak menciptakan asumsi sumber yang tidak dapat dibuktikan.

## 11. Kesimpulan dan Hubungan dengan Task Berikutnya

TASK 02B membuktikan bahwa split awal tidak cukup aman secara visual. TASK 02C menggunakan daftar pasangan terkonfirmasi untuk resplit leakage-safe, lalu meregenerasi review terhadap manifest final.

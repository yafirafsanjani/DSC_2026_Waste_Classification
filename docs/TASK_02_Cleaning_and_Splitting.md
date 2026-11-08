# TASK 02 - Cleaning and Splitting

## 1. Identitas dan Tujuan Task

TASK 02 membangun cleaning deterministik dan split 70/20/10 dari satu-satunya sumber yang diizinkan, `data/extracted/v1/train/`.

## 2. Latar Belakang Masalah

Audit membuktikan bahwa V5 terkontaminasi hubungan dengan V1 dan bahwa V1/test tidak boleh dipakai. Tahap ini harus menjaga provenance, tidak menghapus gambar asli, dan membuat split yang dapat diulang.

## 3. Dataset atau Input yang Digunakan

Input hanya 1.916 file JPG pada `data/extracted/v1/train/`. `cleaning_manifest.csv` dan `split_manifest.csv` menyimpan jalur sumber relatif sehingga provenance dapat diverifikasi.

## 4. Metodologi dan Tahapan Pengerjaan

1. Memindai V1/train saja dan memverifikasi keterbacaan gambar.
2. Mengelompokkan SHA-256 identik; representative leksikografis akan dipilih hanya untuk duplicate dengan label sama.
3. Menghentikan split bila SHA identik atau identitas sumber memiliki label berbeda.
4. Mempertahankan near-duplicate sebagai kandidat manual, bukan menghapusnya otomatis.
5. Melakukan stratified, source-group-aware split dengan seed 42 dan menyalin hasil ke `data/processed/`.
6. Memverifikasi hash salinan, provenance, dan isolasi grup.

## 5. Penjelasan File yang Dibuat atau Dimodifikasi

- `src/data/data_cleaning.py`: validasi V1/train, duplicate decision, manifest cleaning, dan guard provenance.
- `src/data/data_split.py`: allocator split deterministik dan validasi manifest. Implementasi ini kemudian diperluas pada TASK 02C.
- `tests/test_dataset.py`: fixture untuk duplicate, konflik label, group isolation, reproducibility, dan provenance.
- `data/metadata/cleaning_report.csv`, `cleaning_manifest.csv`, `split_manifest.csv`, dan `split_summary.json`: artefak cleaning/splitting.

## 6. Parameter dan Konfigurasi Penting

- Rasio target: train 70%, validation 20%, internal test 10%.
- Random seed: 42.
- Tidak ada augmentasi atau preprocessing pada tahap split.
- Sumber yang diterima secara eksplisit: `data/extracted/v1/train/`.

## 7. Hasil Implementasi dan Artefak

Cleaning mempertahankan seluruh 1.916 gambar. Tidak ada exact duplicate internal, konflik exact-label, atau source identity berulang pada V1/train. Ada 47 kandidat near-duplicate yang dipertahankan untuk review manual.

Split awal menghasilkan train 1.341, validation 384, dan internal test 191. Setiap kelas tetap mendapat target integer stratified: cardboard 183/53/26, electronic 143/41/20, foodwaste 172/49/24, glass 145/41/21, metal 167/48/24, paper 157/45/22, plastic 222/64/32, textile 152/43/22.

## 8. Hasil Pengujian

Pengujian mencakup cleaning deterministik, exact duplicate, konflik lintas label, source-group isolation, split reproducibility, dan provenance. Setelah perluasan sampai TASK 02C, suite regresi final menghasilkan `6 passed`.

## 9. Temuan, Masalah, serta Keterbatasan

Split awal memenuhi hash/source identity isolation, tetapi belum mengisolasi gambar yang berbeda file dan identitas Roboflow namun sangat mirip secara visual. TASK 02B menemukan bahwa kelemahan tersebut menghasilkan kandidat lintas split.

Tidak tersedia sample submission atau konfigurasi lomba lokal yang dapat membuktikan adanya kelas kesembilan. Pipeline tidak membuat kelas baru dan tetap menggunakan delapan folder kelas aktual.

## 10. Keputusan Teknis dan Alasannya

Near-duplicate tidak dihapus otomatis karena perceptual hash dapat menghasilkan false positive. Manifest menyimpan semua gambar valid dan decision logic hanya melakukan deduplikasi untuk SHA identik berlabel sama.

## 11. Kesimpulan dan Hubungan dengan Task Berikutnya

TASK 02 menghasilkan baseline split yang dapat direproduksi, tetapi belum cukup leakage-safe secara visual. TASK 02B melakukan review kandidat visual sebelum resplitting dilakukan pada TASK 02C.

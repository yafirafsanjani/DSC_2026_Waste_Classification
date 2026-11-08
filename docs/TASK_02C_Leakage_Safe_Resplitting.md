# TASK 02C - Leakage-Safe Resplitting

## 1. Identitas dan Tujuan Task

TASK 02C memperbaiki split V1 dengan mengisolasi komponen visual yang telah dikonfirmasi dan mendokumentasikan jejak perubahan. Tujuan utamanya adalah mencegah pasangan visual terkait berada pada split berbeda tanpa menghapus gambar valid.

## 2. Latar Belakang Masalah

TASK 02B menemukan 23 kandidat lintas split pada split awal. Sembilan belas pasangan dinilai berisiko tinggi secara visual. Split lama memenuhi validasi SHA/source identity, namun belum memahami hubungan visual lintas file.

## 3. Dataset atau Input yang Digunakan

Sumber gambar tetap hanya `data/extracted/v1/train/` dengan 1.916 gambar dan delapan kelas. Input metadata mencakup `cleaning_manifest.csv`, split manifest, laporan near-duplicate, serta contact sheet. Tidak ada V1/test atau V5 yang digunakan untuk resplitting.

## 4. Metodologi dan Tahapan Pengerjaan

1. Mengarsipkan manifest dan summary lama dengan SHA-256 sebelum setiap promosi split.
2. Memverifikasi daftar pasangan TASK 02B terhadap manifest aktif.
3. Menggunakan union-find untuk membangun connected component dari 35 pasangan visual yang dikonfirmasi: 19 pasangan lintas split risiko tinggi, 11 pasangan dalam split sama dengan bukti visual kuat, dan empat pasangan tambahan yang menjadi lintas split setelah resplit pertama.
4. Tidak menggabungkan delapan pasangan ambigu atau empat false positive otomatis.
5. Melakukan global group-aware stratified allocation. Komponen multi-label tetap dipindahkan bersama; integritas kelompok diprioritaskan daripada pemisahan per kelas yang sempurna.
6. Membuat salinan baru di `data/.processed_task_02c_staging`, memverifikasi keberadaan dan SHA-256 setiap berkas, lalu menukar direktori processed secara terkendali. Bila staging gagal, split lama dipertahankan.
7. Meregenerasi near-duplicate review terhadap manifest final.

## 5. Penjelasan File yang Dibuat atau Dimodifikasi

- `src/data/data_split.py`: allocator global group-aware, union-find, staging, rollback path, archive checksum, dan validasi final.
- `src/data/near_duplicate_review.py`: digunakan kembali untuk meregenerasi review terhadap manifest baru.
- `tests/test_dataset.py`: mencakup visual group isolation, reproducibility, class representation pada fixture, manifest/file count, provenance, dan gate gagal-aman.
- `data/metadata/near_duplicate_groups.csv`: anggota 33 komponen visual terkonfirmasi.
- `data/metadata/resplitting_report.json`: konfigurasi, daftar pasangan, archive, validasi staging, dan hasil resplit.

## 6. Parameter dan Konfigurasi Penting

- Seed: 42.
- Target rasio: train 70%, validation 20%, internal test 10%.
- Pasangan terkonfirmasi: 35; komponen visual terhubung: 33; anggota komponen: 68.
- Pasangan ambigu: `ND-014`, `ND-015`, `ND-019`, `ND-020`, `ND-021`, `ND-022`, `ND-023`, `ND-025`.
- False positive visual: `ND-003`, `ND-012`, `ND-044`, `ND-046`.

## 7. Hasil Implementasi dan Artefak

Manifest lama awal diarsipkan sebagai `split_manifest_pre_task_02c_e7b830f19c70.csv` dan `split_summary_pre_task_02c_405b0c57e73a.json`. Resplit berikutnya juga menyimpan arsip checksum, sehingga riwayat perubahan tetap tersedia di `data/metadata/*pre_task_02c*`.

Split final tetap berukuran train 1.341, validation 384, dan internal test 191 karena allocator dapat mempertahankan target integer setiap kelas. Per kelas, distribusinya tetap: cardboard 183/53/26, electronic 143/41/20, foodwaste 172/49/24, glass 145/41/21, metal 167/48/24, paper 157/45/22, plastic 222/64/32, textile 152/43/22.

Dibanding manifest awal, 890 dari 1.916 jalur sumber berpindah split untuk mengisolasi kelompok visual. Validasi final menyatakan tidak ada jalur sumber ganda, SHA-256 lintas split, source/visual group lintas split, sumber di luar V1/train, destination tidak valid, file staging hilang, atau mismatch hash salinan.

## 8. Hasil Pengujian

Suite `tests/test_dataset.py` menghasilkan `6 passed`. Test memeriksa audit dasar, duplicate handling, provenance V1/train, split reproducibility, isolasi source group, isolasi visual group, representasi kelas pada fixture, dan validasi gagal-aman.

## 9. Temuan, Masalah, serta Keterbatasan

Semua 35 pasangan visual terkonfirmasi kini berada dalam split yang sama. Namun review final masih memiliki lima pasangan lintas split: `ND-021`, `ND-023`, `ND-025`, `ND-044`, dan `ND-046`. Tiga pertama adalah kandidat ambigu metal dengan distance 4, 4, dan 2; dua terakhir adalah false positive visual sebelumnya pada kelas plastic dengan distance 6. Karena tiga kandidat ambigu masih dapat memengaruhi independensi evaluasi, dataset tidak boleh diklaim bebas leakage secara absolut.

Ada satu komponen visual terkonfirmasi yang memuat label cardboard dan paper (`ND-002`). Ini merupakan indikasi potensi noise label yang tidak diubah otomatis. Komponen tersebut tetap ditempatkan dalam satu split.

## 10. Keputusan Teknis dan Alasannya

Grouping hanya memakai pasangan yang telah dikonfirmasi visual, bukan semua hasil pHash. Keputusan ini mencegah false positive mengubah distribusi secara tidak perlu. Staging dan archive digunakan agar kegagalan validasi tidak menghapus split lama. Perubahan split tidak mengubah gambar asli.

## 11. Kesimpulan dan Hubungan dengan Task Berikutnya

Split final aman terhadap seluruh kelompok visual yang sudah dikonfirmasi, SHA-256, dan source identity. Dataset dapat dipakai untuk eksperimen hanya dengan catatan: tiga pasangan ambigu lintas split dan satu potensi konflik label perlu keputusan manual bila evaluasi yang sangat ketat diperlukan. TASK ini berhenti sebelum preprocessing, augmentasi, atau training.

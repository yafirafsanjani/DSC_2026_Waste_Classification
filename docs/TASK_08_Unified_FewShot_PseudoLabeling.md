# TASK 08: Unified Few-Shot EfficientNet-B2 + Pseudo-Labeling Pipeline (Fully Reproducible Run-All)

**Proyek**: DSC_2026_Waste_Classification  
**Notebook**: 
otebooks/08_fewshot_pseudolabeling_efficientnet_b2.ipynb  
**Dokumen Pendukung**: docs/TASK_08_Unified_FewShot_PseudoLabeling.md  
**Arsitektur Model**: EfficientNet-B2 (Pretrained ImageNet-1K V1)  
**Tipe Pipeline**: Semi-Supervised Few-Shot Learning + Offline Pseudo-Labeling  
**Moda Eksekusi**: **Kernel Restart & Run All (Fully Automated, Deterministic, Single-Notebook)**  

---

## 1. Ringkasan Eksekutif & Tujuan Refactoring Run-All

Eksperimen TASK 08 menyediakan satu berkas notebook Jupyter mandiri (*fully self-contained single notebook*) yang memenuhi ketentuan pengumpulan kompetisi Preliminary Round **Data Science Competition (DSC) Gelar Rasa 2026**:
1. **Single Notebook Requirement**: Seluruh tahapan pengerjaan (audit, preprocessing, CV, teacher training, pseudo-labeling, student training, evaluasi, hingga pembuatan submission CSV) berada dalam satu file 
otebooks/08_fewshot_pseudolabeling_efficientnet_b2.ipynb.
2. **Kernel Restart & Run All Compatible**: Seluruh 10 manual execution flags (RULES_VERIFIED, RUN_FEWSHOT_PREPARATION, dsb.) telah dieliminasi dan digantikan oleh kamus konfigurasi eksperimen terpusat (CONFIG). Juri atau peserta dapat menjalankan seluruh alur secara otomatis dari awal hingga akhir tanpa intervensi manual atau input interaktif.
3. **Kepatuhan Aturan Manual Labeling Ketat**:
   - Kuota manual labeling dibatasi maksimal **5 citra per kelas** (total maksimal **40 citra**) dari 1.920 citra train resmi.
   - Menggunakan berkas anotasi resmi peserta pada: data/manifests/fewshot_pseudolabel_b2_v1/labeled_pool_40.csv.
   - Jika berkas belum diisi, sistem secara otomatis membuat template di manual_annotation_template.csv dan menghentikan eksekusi dengan FileNotFoundError informatif tanpa membuat label palsu atau membaca ground truth lama.
4. **Anti-Leakage & Karantina Tabrakan Data**:
   - 3 citra train yang memiliki kesamaan hash SHA-256 persis dengan citra test kompetisi dikarantina secara otomatis (	rain/1046.jpg == 	est/017.jpg, 	rain/1135.jpg == 	est/201.jpg, 	rain/1821.jpg == 	est/042.jpg).
   - Unlabeled pool berisi 1.877 citra train resmi tanpa kolom label.

---

## 2. Struktur Alur Eksekusi Berurutan (54 Bagian / 108 Cell)

Pipeline terstruktur secara sistematis ke dalam 7 bagian logis (A-G):

### BAGIAN A - PERSIAPAN DATASET & INTEGRITAS (Section 01-11)
- **01. Project Overview**: Deskripsi pipeline, meta-informasi, dan regulasi kompetisi.
- **02. Environment Setup & Centralized Configuration**: Inisialisasi library, seed deterministik (42), deteksi hardware GPU RTX 3050 / CPU fallback, AMP, dan kamus konfigurasi terpusat CONFIG.
- **03. Official Rules Verification & Automated Compliance**: Audit deteksi handbook resmi PDF di docs/ dan penegakan batas aturan secara terprogram (ssert max_labels_per_class <= 5, ssert max_total_labels <= 40).
- **04. Dataset Path Configuration**: Resolusi direktori resmi:
  - data/competition/train/: 1.920 citra tanpa label.
  - data/competition/test/: 480 citra test tanpa label.
  - data/competition/SampleSubmission_DSC.csv.
- **05. Dataset Integrity Audit**: Pemindaian integritas citra, format, dan dimensi.
- **06. Class Definition & Mappings**: 8 kelas target resmi berurutan abjad (cardboard, electronic, oodwaste, glass, metal, paper, plastic, 	extile).
- **07. Labeling Budget Verification**: Fungsi validasi kuota labeling (<= 5 per kelas, <= 40 total).
- **08. Manual Labeled Pool Validation & Ingestion**: Validasi ketersediaan dan keabsahan labeled_pool_40.csv, pembuatan otomatis manual_annotation_template.csv, verifikasi integritas hash anti-leakage.
- **09. Unlabeled Pool Preparation**: Membangun manifest 1.877 citra unlabeled bersih tanpa kolom label dari data train resmi.
- **10. Duplicate and Leakage Audit**: Deteksi 3 citra train yang identik dengan test dan pengamanan karantina.
- **11. Few-Shot Validation Protocol**: Pembuatan 5-Fold Stratified Cross-Validation (32 train, 8 val per fold).

### BAGIAN B - PREPROCESSING & AUGMENTASI (Section 12-17)
- **12. Image Preprocessing**: Transformasi resmi EfficientNet-B2 (Resize 288x288, ImageNet Mean/Std).
- **13. Training Augmentation**: Augmentasi terkendali (Horizontal Flip, Rotasi 15 derajat, Shear 10 derajat, Color Jitter 10%).
- **14. Visualization of Original Images**: Inspeksi visual sampel citra berlabel per kelas.
- **15. Visualization of Augmented Images**: Inspeksi variasi augmentasi stokastik.
- **16. Dataset and DataLoader Construction**: Definisi FewShotLabeledDataset dan UnlabeledDataset (anti-leakage contract).
- **17. Class Distribution Visualization**: Distribusi data berlabel vs unlabeled pool.

### BAGIAN C - FEW-SHOT TEACHER (Section 18-26)
- **18. Teacher Model Initialization**: EfficientNet-B2 Pretrained ImageNet-1K V1 dengan classifier head 8 kelas (1.408 -> 8).
- **19. Model Architecture Summary**: Backbone beku, 11.272 parameter trainable, uji forward pass sintetis (2, 8).
- **20. Few-Shot Cross-Validation Configuration**: Hyperparameter CV (AdamW, lr=1e-3, early stopping patience=5).
- **21. Cross-Validation Execution**: Pelatihan otomatis 5-fold CV atau pemuatan riwayat jika checkpoint telah ada.
- **22. Cross-Validation Results Summary**: Rekapitulasi rata-rata Val Loss, Val Accuracy, dan Val Macro F1.
- **23. Final Teacher Training Configuration**: Parameter pelatihan 40 citra berlabel (15 epoch).
- **24. Final Teacher Training**: Pelatihan otomatis Final Teacher model dan penyimpanan checkpoint persisten.
- **25. Teacher Training History & Curves**: Visualisasi grafik konvergensi pelatihan Teacher.
- **26. Teacher Validation Diagnostics**: Analisis diagnostik bias dan risiko konfirmasi Teacher.

### BAGIAN D - PSEUDO-LABELING (Section 27-33)
- **27. Unlabeled Inference Setup**: Penyiapan DataLoader inferensi pada 1.877 citra unlabeled pool.
- **28. Pseudo-Label Generation**: Inferensi probabilitas softmax Teacher dan penyimpanan ke pseudo_labels.csv.
- **29. Confidence Score Analysis**: Analisis histogram distribusi confidence score.
- **30. Pseudo-Label Filtering**: Penyaringan ketat dengan threshold confidence >= 0.95 ke ccepted_pseudo_labels.csv.
- **31. Class Balance Analysis**: Visualisasi distribusi kelas pseudo-label yang lolos seleksi.
- **32. Pseudo-Label Consistency Audit**: Pemeriksaan konsistensi ID dan jaminan tidak ada kebocoran test.
- **33. Accepted Pseudo-Label Manifest Summary**: Ringkasan jumlah pseudo-label yang diterima.

### BAGIAN E - STUDENT TRAINING (Section 34-41)
- **34. Student Model Initialization**: Inisialisasi model Student EfficientNet-B2 baru.
- **35. Student Dataset Construction**: CombinedStudentDataset yang memadukan 40 citra manual dan N pseudo-label diterima.
- **36. Supervised and Pseudo-Labeled Loss**: SemiSupervisedLoss dengan loss tertimbang (bobot manual 1.0, bobot pseudo 0.5).
- **37. Student Training Configuration**: Hyperparameter Student (AdamW, lr=1e-4, 25 epoch).
- **38. Student Training Execution**: Pelatihan otomatis Student model dan penyimpanan checkpoint persisten.
- **39. Student Training History Summary**: Rekapitulasi tabel riwayat pelatihan epoch per epoch.
- **40. Student Learning Curves**: Plot grafik loss dan akurasi Student.
- **41. Student Evaluation**: Evaluasi model Student pada himpunan data berlabel manual.

### BAGIAN F - EVALUASI KOMPARATIF (Section 42-47)
- **42. Teacher vs Student Comparison**: Tabel komparasi performa dan peran Teacher vs Student.
- **43. Validation Metrics**: Rekapitulasi metrik generalisasi dari 5-fold cross-validation.
- **44. Classification Report**: Laporan presisi, recall, dan F1-score untuk 8 kelas target.
- **45. Confusion Matrix**: Visualisasi matriks kontingensi mentah dan ternormalisasi.
- **46. Overfitting / Underfitting Analysis**: Kajian diagnostik transfer learning pada rejim few-shot.
- **47. Experiment Summary**: Rekapitulasi artefak checkpoint dan manifest.

### BAGIAN G - KAGGLE SUBMISSION (Section 48-54)
- **48. Competition Test Audit**: Verifikasi 480 citra test kompetisi dan kesesuaian dengan SampleSubmission_DSC.csv.
- **49. Standalone Inference Configuration**: Setup inferensi mandiri tanpa ketergantungan variabel pelatihan.
- **50. Best Model Selection**: Seleksi otomatis checkpoint terbaik (Student jika ada, Teacher sebagai fallback).
- **51. Competition Test Inference**: Inferensi otomatis pada 480 citra test resmi.
- **52. Submission CSV Generation**: Pembuatan berkas Final Dataset for Submission to Kaggle/submission_fewshot_pseudolabel_b2.csv.
- **53. Submission CSV Verification**: Verifikasi otomatis: tepat 480 baris, format ID leading-zero utuh, 8 kelas valid.
- **54. Final Submission Summary & Kaggle Upload Guide**: Panduan verifikasi akhir dan pengunggahan manual peserta.

---

## 3. Kamus Konfigurasi Terpusat (CONFIG)

Konfigurasi terletak di Section 02 notebook:

`python
CONFIG = {
    "experiment_id": "fewshot_pseudolabel_b2_v1",
    "random_seed": 42,
    "image_size": 288,
    "batch_size": 8,
    "num_workers": 0,
    "max_labels_per_class": 5,
    "max_total_labels": 40,
    "cv_folds": 5,
    "teacher_epochs": 15,
    "teacher_lr": 1e-3,
    "teacher_patience": 5,
    "pseudo_label_confidence_threshold": 0.95,
    "pseudo_label_loss_weight": 0.5,
    "student_epochs": 25,
    "student_lr": 1e-4,
    "student_patience": 7,
    "mixed_precision": USE_AMP,
    "device": str(DEVICE)
}
`

---


---

## 5. Alat Anotasi Manual Mandiri (`tools/manual_fewshot_annotation.py`)

Untuk mempermudah peserta dalam memilih dan memberi label pada **tepat 5 citra per kelas** (total 40 citra) dari 1.920 data train resmi tanpa ketergantungan pada template statis 40 citra pertama, telah disediakan utilitas anotasi visual mandiri:

### Fitur Utama:
1. **Desktop GUI Modern (Tkinter + Pillow)**: Menampilkan citra kandidat secara visual dengan resolusi terjaga, nomor indeks, dan rincian metadata.
2. **Karantina Otomatis Anti-Leakage**: 3 citra train yang memiliki kesamaan hash SHA-256 persis dengan citra test (`1046.jpg`, `1135.jpg`, `1821.jpg`) secara otomatis dikarantina dan tidak pernah ditampilkan kepada peserta.
3. **Penegakan Kuota Real-Time**:
   - Menghitung jumlah anotasi per kelas secara langsung (misal `cardboard: 3/5`).
   - Begitu suatu kelas mencapai 5 sampel, tombol kelas tersebut otomatis terkunci (**DISABLED**) untuk mencegah kelebihan kuota.
   - Total anotasi dibatasi tepat 40 sampel.
4. **Navigasi Lengkap**:
   - `[S] Skip`: Melewati citra saat ini tanpa memberi label.
   - `[U] Undo`: Membatalkan anotasi terakhir dan mengembalikan kuota.
   - `[R] Reset Skips`: Mengembalikan citra-citra yang sempat dilewati ke dalam rotasi kandidat.
   - `[E] Ekspor CSV`: Mengekspor berkas jika 40 sampel telah lengkap.
5. **Penyimpanan Progres Persisten**:
   - Setiap aksi tersimpan otomatis ke `data/manifests/fewshot_pseudolabel_b2_v1/annotation_progress.json`.
   - Jika aplikasi ditutup di tengah jalan, progres tidak akan hilang dan dapat dilanjutkan kapan saja.
6. **Ekspor Otomatis & Terverifikasi**:
   - Ketika 40 sampel tercapai (5 sampel untuk setiap 8 kelas), sistem secara otomatis menghasilkan dan memverifikasi:
     `data/manifests/fewshot_pseudolabel_b2_v1/labeled_pool_40.csv`
   - Format CSV memuat kolom: `image_id,class_name,class_idx,relative_path,sha256` yang 100% kompatibel dengan Section 08 notebook final.

### Cara Menjalankan:
```powershell
# 1. Jalankan Antarmuka Visual Desktop (GUI Rekomendasi):
.env_gelarrasa\Scripts\python.exe tools/manual_fewshot_annotation.py

# 2. Atau Jalankan Mode Terminal Konsol (CLI):
.env_gelarrasa\Scripts\python.exe tools/manual_fewshot_annotation.py --cli

# 3. Cek Status Progres Anotasi:
.env_gelarrasa\Scripts\python.exe tools/manual_fewshot_annotation.py --info
```

## 4. Prasyarat & Panduan Eksekusi

### Prasyarat Pengisian Labeled Pool:
Sebelum Kernel Restart & Run All berhasil dieksekusi hingga akhir, peserta wajib menyediakan berkas anotasi manual:
1. Jalankan Section 01-08 sekali (atau amati saat Run All berhenti di Section 08).
2. Sistem otomatis menghasilkan template di:
   data/manifests/fewshot_pseudolabel_b2_v1/manual_annotation_template.csv
3. Isi kolom class_name pada 40 citra train tersebut (tepat 5 citra per kelas untuk 8 kelas resmi).
4. Simpan file yang telah diisi ke:
   data/manifests/fewshot_pseudolabel_b2_v1/labeled_pool_40.csv
5. Lakukan **Kernel Restart & Run All**. Seluruh alur (Teacher, Pseudo-Labeling, Student, Submission CSV) akan berjalan otomatis hingga tuntas.

### Penanganan Pretrained Weights:
Bobot pretrained ImageNet-1K diunduh oleh torchvision secara otomatis ke cache Torch (~/.cache/torch/hub/checkpoints/) pada saat pertama kali inisialisasi, dan digunakan secara lokal tanpa ketergantungan API pihak ketiga.

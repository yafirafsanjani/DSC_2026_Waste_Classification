# TASK 08: Unified Few-Shot EfficientNet-B2 + Pseudo-Labeling Pipeline

**Proyek**: `DSC_2026_Waste_Classification`  
**Notebook**: `notebooks/08_fewshot_pseudolabeling_efficientnet_b2.ipynb`  
**Dokumen Pendukung**: `docs/TASK_08_Unified_FewShot_PseudoLabeling.md`  
**Peran**: Senior Computer Vision Engineer, Machine Learning Engineer, dan MLOps Engineer  

---

## 1. Ringkasan Eksekutif & Tujuan Eksperimen

Eksperimen TASK 08 mengimplementasikan pipeline klasifikasi citra semi-supervised mandiri (*fully self-contained*) dalam **satu berkas notebook tunggal** (`notebooks/08_fewshot_pseudolabeling_efficientnet_b2.ipynb`). Pipeline ini dirancang khusus untuk mematuhi regulasi kompetisi baru dengan skenario **Few-Shot Learning**:

1. **Batasan Anggaran Label Manual**: Maksimal **5 citra per kelas** atau total **maksimal 40 citra manual** untuk 8 kelas target.
2. **Sisa Dataset Training**: Dialokasikan sebagai **Unlabeled Pool** (~1.876 citra) tanpa akses terhadap ground-truth label asli.
3. **Data Test Kompetisi**: Sebanyak 480 citra pada `data/competition/test/` hanya digunakan untuk inferensi akhir (*submission generation*), tanpa kebocoran data (*zero leakage*).
4. **Metodologi**: Paradigma Teacher-Student Pseudo-Labeling berbasis backbone **EfficientNet-B2 pretrained ImageNet-1K**, disempurnakan dengan 5-Fold Stratified Cross-Validation, penyaringan keyakinan tinggi (*high-confidence filtering* $\ge 0.95$), dan *weighted semi-supervised loss*.

Notebook ini disiapkan untuk dikumpulkan langsung kepada panitia/juri lomba, sehingga **seluruh fungsi inti diimplementasikan secara mandiri di dalam notebook tanpa ketergantungan pada modul `src/` proyek**.

---

## 2. Status Verifikasi Regulasi & Approval Gate

- Dokumen regulasi resmi belum ditemukan di direktori `docs/` workspace.
- Oleh karena itu, notebook menerapkan **manual approval gate** dengan flag `RULES_VERIFIED = False` secara default.
- Pipeline menegakkan 3 pilar kepatuhan:
  - Kuota pelabelan manual $\le 5$ sampel per kelas ($\le 40$ sampel total) diverifikasi melalui validator otomatis (`verify_labeling_budget`).
  - Larangan pembacaan label test resmi atau label sample submission sebagai target.
  - Pseudo-labeling murni dibangkitkan dari model Teacher yang sah tanpa interpolasi ground truth lama.

---

## 3. Dataset Management & Arsitektur Partisi

### 3.1 Sumber Data
- **Raw Training Images**: `data/extracted/v1/train` (1.916 citra pada 8 subfolder kelas).
- **Competition Test Images**: `data/competition/test` (480 citra tanpa label).
- **Sample Submission Template**: `data/competition/SampleSubmission_DSC.csv`.

### 3.2 Partisi & Isolasi Anti-Leakage
| Partisi | Jumlah Sampel | Peran | Proteksi Kebocoran |
| :--- | :---: | :--- | :--- |
| **Labeled Pool** | 40 citra | Pelatihan Teacher & Student (5/kelas) | Diverifikasi kuota ketat, hash SHA-256 dicatat |
| **Unlabeled Pool** | ~1.876 citra | Target Pseudo-Labeling | Kolom `class_name` & `class_idx` dihapus permanen |
| **CV Folds** | 5 fold (32 train, 8 val) | Estimasi performa Out-Of-Fold Teacher | Folds terisolasi 100% tanpa tumpang tindih |
| **Competition Test** | 480 citra | Inferensi Submission Kaggle | Read-only hash audit, 0 overlap dengan train |

---

## 4. Spesifikasi Preprocessing & Augmentasi

### 4.1 Standar Preprocessing EfficientNet-B2 (Evaluasi & Inferensi)
Mengikuti metadata bobot resmi `EfficientNet_B2_Weights.IMAGENET1K_V1`:
- Format: RGB Tensor float32 skala $[0.0, 1.0]$
- Resize: Resolusi 288 (Bicubic Interpolation, antialias=True)
- Center Crop: $288 	imes 288$ piksel
- Normalisasi ImageNet:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

### 4.2 Augmentasi Pelatihan Terkendali
Untuk rejim few-shot, augmentasi mencegah *memorization* tanpa merusak karakteristik semantik kelas:
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=15)`
- `RandomAffine(degrees=0, shear=10)`
- `ColorJitter(brightness=0.10)`
- Resize 288 & Center Crop $288 	imes 288$ dengan Normalisasi ImageNet.

---

## 5. Arsitektur Model: Teacher & Student

### 5.1 EfficientNet-B2 Backbone
- **Model Dasar**: `torchvision.models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)`
- **Modifikasi Classifier**: Linear layer akhir diubah dari $1408 ightarrow 8$ kelas, mempertahankan layer Dropout (0.3).
- **Kebijakan Pembekuan (Freezing)**: Seluruh layer ekstraksi fitur (`features.parameters()`) dibekukan untuk mengisolasi gradien hanya pada classifier head:
  - Total Parameter: **7.712.266**
  - Parameter Trainable: **11.272** (Head linear layer: $1408 	imes 8 + 8$)
  - Parameter Frozen: **7.700.994**

### 5.2 Model Student & Semi-Supervised Loss
Model Student diinisialisasi dari EfficientNet-B2 pretrained ImageNet dan dilatih menggunakan dataset gabungan (`CombinedStudentDataset`):
- Data Supervised: 40 sampel manual (bobot sampel = 1.0).
- Data Pseudo-Labeled: Sampel dengan keyakinan $\ge 0.95$ (bobot loss $\lambda_{pseudo} = 0.5$).
- **Fungsi Loss Gabungan**:
  $$\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda_{pseudo} \cdot \mathcal{L}_{pseudo}$$

---

## 6. Training Monitoring & Checkpoint Management

### 6.1 Monitoring Terintegrasi
- **Batch Progress**: Progress bar tunggal berbasis `tqdm.auto` yang kompatibel dengan Jupyter, menampilkan epoch, batch, running loss, running accuracy, dan learning rate.
- **Epoch Summary**: Mencatat train loss, train acc, val loss, val acc, macro precision, recall, macro F1, learning rate, durasi epoch, counter early stopping, dan status penyimpanan checkpoint.
- **Persistent Logging**: Riwayat disimpan dalam format JSON terstruktur pada `outputs/metrics/`.

### 6.2 Checkpoint Namespace: `fewshot_pseudolabel_b2_v1`
- `models/checkpoints/fewshot_pseudolabel_b2_v1/cv/fold_{i}/best.pt` (Cross-Validation per fold)
- `models/checkpoints/fewshot_pseudolabel_b2_v1/teacher/best.pt` (Final Teacher pada 40 sampel)
- `models/checkpoints/fewshot_pseudolabel_b2_v1/student/best.pt` (Final Student model)

---

## 7. Standalone Inference & Kaggle Submission

### 7.1 Fresh-Kernel Standalone Inference (Section 49-53)
Bagian inferensi dapat dieksekusi secara independen pada kernel baru tanpa memerlukan state variabel dari cell pelatihan sebelumnya:
1. Menentukan `PROJECT_ROOT` secara dinamis.
2. Memuat arsitektur EfficientNet-B2 dengan `weights=None` dan memuat `state_dict` langsung dari checkpoint disk.
3. Menyiapkan transform evaluasi resmi (288x288, ImageNet norm).
4. Memvalidasi keberadaan 480 citra test dan membaca template `SampleSubmission_DSC.csv`.

### 7.2 Spesifikasi Berkas Submission CSV
- **Path Target**: `Final Dataset for Submission to Kaggle/submission_fewshot_pseudolabel_b2.csv`
- **Format**: Dua kolom resmi `image_id,waste_category`
- **Integritas**:
  - Tepat 480 baris data (plus 1 header baris).
  - Format leading zero terjaga (`001`, `002`, dst.).
  - 0 nilai NaN/null/kosong.
  - Semua prediksi berada dalam 8 kelas resmi kompetisi.
  - Urutan baris identik 100% dengan sample submission.

---

## 8. Panduan Eksekusi Manual Langkah demi Langkah

Pengguna menjalankan eksperimen secara manual dengan mengaktifkan flag pada notebook `notebooks/08_fewshot_pseudolabeling_efficientnet_b2.ipynb`:

| Langkah | Section Notebook | Flag yang Diaktifkan | Aksi yang Dijalankan |
| :---: | :--- | :--- | :--- |
| **1** | 01 ? 07 | `RULES_VERIFIED = True` (opsional) | Inisialisasi environment, audit regulasi, audit integritas |
| **2** | 08 ? 11 | `RUN_FEWSHOT_PREPARATION = True` | Seleksi 40 sampel labeled, pembuatan unlabeled pool & CV folds |
| **3** | 12 ? 17 | *(tanpa flag)* | Inisialisasi transform, preview augmentasi, konstruksi DataLoader |
| **4** | 18 ? 22 | `RUN_CROSS_VALIDATION = True` | Eksekusi 5-Fold Cross-Validation Teacher model |
| **5** | 23 ? 26 | `RUN_TEACHER_TRAINING = True` | Pelatihan Final Teacher pada seluruh 40 citra berlabel |
| **6** | 27 ? 33 | `RUN_PSEUDO_LABEL_GENERATION = True`<br>`RUN_PSEUDO_LABEL_SELECTION = True` | Inferensi unlabeled pool & penyaringan pseudo-label (th $\ge 0.95$) |
| **7** | 34 ? 41 | `RUN_STUDENT_TRAINING = True` | Pelatihan model Student pada data gabungan |
| **8** | 42 ? 47 | `RUN_TEACHER_STUDENT_EVALUATION = True` | Evaluasi komparatif, matriks konfusi, dan ringkasan |
| **9** | 48 ? 54 | `RUN_TEST_INFERENCE = True`<br>`RUN_SUBMISSION_GENERATION = True` | Inferensi test 480 citra, pembuatan CSV, dan verifikasi akhir |

---

## 9. Hasil Pengujian Mutu (QA & Synthetic Tests)

Suite pengujian otomatis pada `tests/test_fewshot_pseudolabel_b2.py` memverifikasi 10 kriteria kualitas dengan status **100% PASS**:
1. `test_notebook_file_and_cells_structure`: Tepat 108 cell (54 markdown, 54 code).
2. `test_notebook_code_cells_syntax_compilation`: 54/54 code cell lulus kompilasi sintaks Python tanpa error.
3. `test_manual_execution_flags_default_false`: Seluruh 10 manual flags bernilai `False` secara default.
4. `test_labeling_budget_validator`: Menolak dataset $>5$ per kelas atau $>40$ total sampel.
5. `test_unlabeled_dataset_anti_leakage_contract`: Data unlabeled terbukti tidak mengembalikan true label.
6. `test_stratified_cv_splits_disjointness`: Folds CV terisolasi sempurna (32 train, 8 val per fold).
7. `test_teacher_architecture_modification_and_forward_pass`: Arsitektur EfficientNet-B2 8 kelas, 11.272 parameter trainable.
8. `test_student_semi_supervised_loss`: Formulasi loss semi-supervised terbobot tervalidasi.
9. `test_standalone_fresh_kernel_setup_logic`: Setup inferensi kernel baru bekerja mandiri.
10. `test_submission_verification_rules`: Skema submission 480 baris tervalidasi ketat.

---

## 10. Analisis Risiko & Keterbatasan

1. **Varians Rejim Few-Shot**: 40 sampel manual memberikan varians estimasi yang relatif tinggi pada evaluasi validasi.
2. **Confirmation Bias**: Jika Teacher melakukan misklasifikasi dengan keyakinan tinggi, error tersebut dapat diserap oleh model Student. Mitigasi dilakukan dengan menetapkan ambang batas tinggi ($0.95$) dan bobot loss pseudo yang lebih rendah ($0.5$).
3. **Keterbatasan Generalisasi Tanpa Label Tambahan**: Model semi-supervised sangat bergantung pada representasi ImageNet pretrained.

*Catatan: Dokumen ini disusun sebagai panduan teknis implementasi TASK 08. Seluruh proses training dan inferensi sesungguhnya dijalankan secara manual oleh pengguna.*

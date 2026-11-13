# TASK 09 — Optimized Few-Shot Classification & Semi-Supervised Learning (SSL) V2

**Proyek**: Data Science Competition (DSC) Gelar Rasa 2026 — Waste Classification  
**Notebook Utama**: `notebooks/09_optimized_fewshot_ssl.ipynb`  
**Target Submission**: `Final Dataset for Submission to Kaggle/submission_optimized_fewshot_ssl.csv`  
**Baseline Pembanding (TASK 08)**: Public Leaderboard Kaggle **0.5902**

---

## 1. Analisis Akar Masalah Baseline TASK 08

Pada eksperimen TASK 08, pipeline menghasilkan skor baseline 0.5902 pada platform Kaggle. Analisis forensik menemukan 3 akar masalah teknis utama:

1. **Degradasi Statistik BatchNorm pada Backbone yang Dibekukan**:
   - Di TASK 08, walaupun `freeze_backbone=True` membekukan gradien parameter (`requires_grad=False`), fungsi pelatihan memanggil `model.train()`.
   - Secara default di PyTorch, `model.train()` mengatur seluruh submodule (termasuk layer `nn.BatchNorm2d` di dalam backbone) ke mode `training=True`.
   - Akibatnya, pada setiap forward pass dengan batch size kecil (8) pada dataset 32 citra train fold, buffer `running_mean` dan `running_var` terus diperbarui dan bergeser menjauh dari distribusi representasi ImageNet.
   - Saat evaluasi (`model.eval()`), representasi fitur terdegradasi.

2. **Threshold Pseudo-Labeling yang Terlalu Kaku (0.95)**:
   - Softmax pada 8 kelas dari classifier few-shot yang ter-regularisasi menghasilkan skor kepercayaan (*confidence*) maksimum sekitar 0.705.
   - Ambang batas kaku 0.95 menyebabkan **0 citra pseudo-label yang lolos**, sehingga pelatihan model Student tidak dapat memanfaatkan 1.877 citra unlabeled yang tersedia.

3. **Overfitting pada Neural Network Classifier Head (N=40)**:
   - Mengoptimalkan 11.272 parameter linear head menggunakan AdamW pada hanya 32 citra train per fold rentan menghafal (*memorizing*) sampel latihan (train accuracy mencapai 100%, sementara loss validasi meningkat hingga ~1.5).

---

## 2. Rincian Solusi & Peningkatan pada TASK 09

### A. Strict BatchNorm Freezing
- Mengimplementasikan fungsi `freeze_backbone_with_eval_bn(model)`:
  - Membekukan seluruh bobot konvolusi (`param.requires_grad = False`).
  - Memaksa seluruh layer `nn.BatchNorm2d` dan `nn.BatchNorm1d` tetap berada dalam mode `eval()` (`m.eval()`).
- Melalui unit test `test_batchnorm_freeze_protects_running_statistics`, terbukti bahwa `running_mean` dan `running_var` 100% stabil dan tidak berubah selama pelatihan classifier head.

### B. Evaluasi 4 Kandidat Classifier Few-Shot (5-Fold CV OOF)
- Mengekstrak 1.408-dimensional L2-normalized embeddings dari backbone EfficientNet-B2 pretrained ImageNet-1K.
- Menguji 4 metode klasifikasi pada 5-Fold Stratified Cross-Validation yang identik:
  1. **PyTorch Linear Head (Frozen BN)**: AdamW, lr=1e-3, weight_decay=1e-2.
  2. **Logistic Regression (L2)**: Regulasi cembung $C=1.0$, solver L-BFGS.
  3. **Linear Support Vector Classifier (Linear SVC)**: Pemaksimal margin $C=1.0$ dengan kalibrasi probabilitas Platt scaling.
  4. **Cosine Prototype Classifier**: Non-parametrik, menghitung centroid tiap kelas yang dinormalisasi L2, immune terhadap overfitting.
- Memilih model terbaik secara objektif berdasarkan **Out-Of-Fold (OOF) Macro F1 Score**.

### C. Safe & Calibrated Pseudo-Labeling
- **Two-View Augmentation Consistency**:
  - Mengevaluasi setiap citra unlabeled dalam 2 pandangan: Citra Asli (View 1) dan Horizontal Flip (View 2).
  - Hanya menerima citra yang menghasilkan kelas prediksi identik (`pred_v1 == pred_v2`).
- **Confidence Floor Terukur**:
  - Menetapkan batas bawah `pseudo_min_confidence = 0.45` (jauh di atas tebakan acak 8 kelas = 0.125).
- **Class-Balanced Top-K Filtering**:
  - Membatasi paling banyak $K=15$ citra dengan confidence tertinggi per kelas.
  - Mencegah bias kelas dominan dan menjamin keragaman representasi.
  - Seluruh pseudo-label berasal dari 1.877 citra train unlabeled (anti-leakage: data test tidak disentuh).

### D. Student Training
- Jika pseudo-label valid tersedia:
  - Dataset = 40 sampel manual ($w_{\text{sup}} = 1.0$) + $M$ sampel pseudo-label ($w_{\text{pseudo}} = 0.35$).
  - Fungsi Loss Semi-Supervised: $\mathcal{L} = \mathcal{L}_{\text{sup}} + 0.35 \mathcal{L}_{\text{pseudo}}$.
  - Dilatih menggunakan Mixed Precision (AMP) dan backbone BatchNorm yang dibekukan.
- Jika pseudo-label tidak mencukupi:
  - Pipeline secara otomatis melakukan fallback transparan ke model Final Teacher terbaik.

### E. Test-Time Augmentation (TTA)
- Inferensi pada 480 citra test kompetisi menggabungkan prediksi dari citra asli dan citra yang dibalik secara horizontal:
  $$p_{\text{TTA}}(x) = \frac{1}{2} \left( p(x) + p(\text{flip}(x)) \right)$$
- Mendukung berkas multi-ekstensi (`477 .jpg`, `2 .jpeg`, `1 .png`) tanpa mengacak urutan ID.

---

## 3. Struktur Notebook `09_optimized_fewshot_ssl.ipynb`

Notebook terdiri atas 22 Section (44 cell total: 22 Markdown, 22 Code):

1. **01. Project Setup & Environment**: Resolusi path, seed 42, deteksi RTX 3050, AMP.
2. **02. Experiment Configuration**: Konfigurasi terpusat `CONFIG`.
3. **03. Official Rules & Labeling Budget Verification**: Pemeriksaan kuota 40 label dan kepatuhan aturan.
4. **04. Dataset Loading & Flat Structure Integrity**: Pemuatan `labeled_pool_40.csv` dan `unlabeled_pool.csv`.
5. **05. Stratified 5-Fold Cross-Validation Alignment**: Penyelarasan split dengan `cv_folds.csv`.
6. **06. Preprocessing & Augmentation Pipelines**: Transformasi resmi ImageNet 288x288 dan weak augmentation.
7. **07. Pretrained Backbone Initialization & BatchNorm Freeze Fix**: Perbaikan BatchNorm dan diagnostik mandiri.
8. **08. Frozen Feature Extraction & Embedding Caching**: Caching matriks fitur L2-normalized 40 sampel.
9. **09. Few-Shot Classifiers 5-Fold CV Out-Of-Fold Evaluation**: Evaluasi 4 classifier pada 5 fold.
10. **10. OOF Evaluation Metrics & Model Selection**: Tabel perbandingan OOF, matriks konfusi, seleksi model terbaik.
11. **11. Final Supervised Model Training on Full 40 Labeled Pool**: Fitting model terpilih pada seluruh 40 label.
12. **12. Unlabeled Feature Extraction & Multi-View Augmentation**: Ekstraksi fitur 2-view pada 1.877 citra unlabeled.
13. **13. Pseudo-Label Generation & Confidence Distribution Analysis**: Analisis sebaran confidence skor.
14. **14. Multi-View Consistency & Class-Balanced Top-K Filtering**: Penyaringan pseudo-label berkualitas.
15. **15. Student Dataset Construction & SemiSupervisedLoss**: Dataset gabungan berbobot.
16. **16. Student Model Training with Proper BatchNorm Freezing**: Pelatihan Student model dengan AMP.
17. **17. Student vs Teacher Evaluation & Final Model Selection**: Seleksi model definitif untuk inferensi test.
18. **18. Test-Time Augmentation (TTA) Formulation**: Formulasi ensembling probabilitas TTA.
19. **19. Competition Test Inference (480 Samples Multi-Extension)**: Inferensi test 480 citra multi-format.
20. **20. Submission CSV Generation & Verification**: Ekspor dan verifikasi ketat file CSV submission.
21. **21. Performance Diagnostic & Comparison with Baseline (0.5902)**: Rangkuman perbandingan teknis.
22. **22. Kaggle Upload Guide & Instructions**: Panduan pengunggahan hasil ke platform Kaggle.

---

## 4. Cara Menjalankan Notebook

Pengguna dapat menjalankan eksperimen secara mandiri melalui salah satu cara berikut:

### Opsi A: Eksekusi Penuh (Restart Kernel & Run All)
1. Buka Jupyter Notebook pada browser.
2. Buka berkas `notebooks/09_optimized_fewshot_ssl.ipynb`.
3. Pilih menu **Kernel** → **Restart Kernel and Run All Cells**.
4. Seluruh proses (caching fitur, CV OOF, seleksi model, pseudo-labeling, inferensi TTA, dan pembuatan submission) akan berjalan otomatis hingga selesai tanpa interupsi.

### Opsi B: Eksekusi Bertahap (Cell by Cell)
1. Jalankan Section 01–06 untuk inisialisasi lingkungan dan dataset.
2. Jalankan Section 07–10 untuk melihat perbandingan performa 4 arsitektur classifier pada 5-Fold CV.
3. Jalankan Section 11–14 untuk menghasilkan dan menyaring pseudo-label.
4. Jalankan Section 15–17 untuk pelatihan Student dan seleksi model terbaik.
5. Jalankan Section 18–22 untuk menghasilkan file submission `Final Dataset for Submission to Kaggle/submission_optimized_fewshot_ssl.csv`.

---

## 5. Perbaikan PyTorch 2.6+ / 2.10 UnpicklingError & Checkpoint Architecture Compatibility

### Akar Masalah:
Pada PyTorch versi 2.6 ke atas (termasuk PyTorch 2.10.0+cu128 di lingkungan proyek), parameter `weights_only` pada `torch.load()` secara default bernilai `True` untuk alasan keamanan.
Ketika model Teacher (`LogisticRegression_L2`) disimpan pada Section 11, metadata `meta` memuat objek NumPy array (`numpy._core.multiarray._reconstruct` / `numpy.ndarray` dari `clf.coef_` dan `clf.intercept_`). Akibatnya, pemanggilan `torch.load()` di Section 19 memicu `_pickle.UnpicklingError: Weights only load failed. Unsupported global: numpy._core.multiarray._reconstruct`.

### Solusi Permanen:
1. **Fungsi Pemuatan Aman `load_project_checkpoint(path, device)`**:
   - Memvalidasi secara ketat bahwa target berkas berada di dalam direktori proyek (`PROJECT_ROOT`). Akses berkas di luar proyek langsung ditolak (`PermissionError`).
   - Mencoba pemuatan dengan `weights_only=True` terlebih dahulu.
   - Menyediakan fallback terkontrol ke `weights_only=False` secara eksplisit HANYA jika berkas terbukti merupakan checkpoint sah internal proyek.
2. **Sanitasi Metadata Checkpoint Menjadi Tensor Murni**:
   - Section 11 diperbarui agar secara otomatis mengonversi seluruh parameter NumPy (`coef_`, `intercept_`) menjadi `torch.Tensor` sebelum disimpan ke `best.pt`.
   - File checkpoint `models/checkpoints/fewshot_ssl_v2/teacher/best.pt`, `labeled_embeddings_v2.pt`, dan `unlabeled_embeddings_v2.pt` yang sudah ada di disk telah disanitasi sehingga kini kompatibel 100% dengan `weights_only=True`.
3. **Fleksibilitas Arsitektur Classifier pada Section 19**:
   - Model yang terpilih pada eksperimen 5-Fold CV OOF adalah `LogisticRegression_L2`.
   - `PyTorchLinearWrapper` dilengkapi metode `load_head_state_dict(state_dict)` yang fleksibel memuat bobot baik dengan prefix `fc.` maupun tanpa prefix.
   - Pemuatan inferensi TTA Section 19 langsung mendukung classifier yang terpilih tanpa ketergantungan variabel memori sesi sebelumnya.

### Fresh-Kernel Standalone Inference (Section 19–22):
Section 19 telah direkayasa secara khusus agar dapat dieksekusi secara mandiri dari kernel Jupyter yang baru/bersih tanpa mengeksekusi ulang Section 01–18:
1. Memuat seluruh import mandiri (`Path`, `torch`, `v2`, `pandas`, `Image`, dll) di awal cell.
2. Merekonsiliasi `PROJECT_ROOT`, perangkat `DEVICE` (CUDA/CPU), pemetaan 8 kelas resmi, dan `DATA_TEST_DIR`.
3. Memuat `eval_transforms`, `weak_flip_transforms`, `CompetitionTestTTADataset`, dan `PyTorchLinearWrapper`.
4. Menggunakan `load_project_checkpoint` dengan `weights_only=True` pada checkpoint `models/checkpoints/fewshot_ssl_v2/teacher/best.pt`.
5. Mendukung rekonstruksi parameter `LogisticRegression_L2` secara presisi matematis ke linear layer PyTorch untuk inferensi TTA.
6. Memungkinkan pengguna langsung menghasilkan submission CSV dari Section 19 -> Section 20 -> Section 21 -> Section 22 dalam hitungan detik.

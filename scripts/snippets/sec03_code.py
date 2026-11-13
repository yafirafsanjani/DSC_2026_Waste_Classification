# ==============================================================================
# SECTION 03: OFFICIAL RULES VERIFICATION & APPROVAL GATE
# ==============================================================================
def resolve_project_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent
    ]
    for candidate in candidates:
        if (candidate / "data" / "competition").exists():
            return candidate.resolve()
    return Path.cwd().resolve()

if "PROJECT_ROOT" not in globals():
    PROJECT_ROOT = resolve_project_root()

def detect_competition_handbook(docs_dir: Path) -> List[Dict[str, Any]]:
    # Mencari berkas PDF handbook/regulasi secara case-insensitive pada direktori docs
    matched_files = []
    if not docs_dir.exists():
        return matched_files
        
    keywords = ["handbook", "preliminary", "dsc", "gelar", "rasa", "rule", "regulasi", "panduan", "guide", "ketentuan", "pedoman"]
    
    for item in sorted(docs_dir.iterdir(), key=lambda p: p.name.lower()):
        if item.is_file() and item.suffix.lower() == ".pdf":
            name_lower = item.name.lower()
            if any(kw in name_lower for kw in keywords):
                matched_files.append({
                    "name": item.name,
                    "path": item.resolve(),
                    "relative_path": item.relative_to(PROJECT_ROOT).as_posix(),
                    "size_bytes": item.stat().st_size
                })
    return matched_files

DOCS_DIR = PROJECT_ROOT / "docs"
detected_handbooks = detect_competition_handbook(DOCS_DIR)

print("=" * 75)
print("HASIL AUDIT DETEKSI DOKUMEN REGULASI RESMI:")
print(f"Direktori Pencarian   : {DOCS_DIR.resolve()}")
print(f"Dokumen Ditemukan     : {len(detected_handbooks) > 0}")

if detected_handbooks:
    for idx, doc_info in enumerate(detected_handbooks, 1):
        print(f"\\n[DOKUMEN {idx}]")
        print(f"  Nama Berkas   : {doc_info['name']}")
        print(f"  Path Lengkap  : {doc_info['path']}")
        print(f"  Ukuran Berkas : {doc_info['size_bytes'] / (1024*1024):.2f} MB")
else:
    print("  [PERINGATAN] Berkas PDF regulasi/handbook tidak ditemukan di docs/.")

# Ringkasan Eksekutif Aturan Resmi Berdasarkan Dokumen (Handbook Halaman 7-8)
OFFICIAL_RULES_SUMMARY = {
    "hand_labeling_train_limit": "Maksimal 5 gambar per kelas (total 40 gambar manual dari 1920 data train).",
    "test_data_restriction": "Dilarang hand labeling pada data test dan dilarang menggunakan data test untuk training.",
    "pretrained_model_allowed": "Pretrained model diperbolehkan selama dijalankan dalam lingkungan lokal peserta.",
    "external_dataset_prohibited": "Dilarang menggunakan dataset eksternal; hanya boleh memakai dataset panitia.",
    "automl_llm_genai_prohibited": "Dilarang memakai AutoML/LLM, VLM, Generative AI, atau API pihak ketiga untuk model/prediksi.",
    "pseudo_labeling_status": "Tidak dijelaskan secara eksplisit di handbook (digunakan sebagai strategi teknis mandiri).",
    "single_notebook_reproducible": "Wajib 1 file notebook (.ipynb) yang reproducible dan menghasilkan skor sama dengan CSV submission.",
    "max_submissions_per_day": 3,
    "submission_deadline": "14 November 2026 pukul 23.59 WIB"
}

print("\\n" + "=" * 75)
print("RINGKASAN ATURAN KOMPETISI RESMI (HANDBOOK HALAMAN 7-8):")
print(f"1. Kuota Hand Labeling Data Train : {OFFICIAL_RULES_SUMMARY['hand_labeling_train_limit']}")
print(f"2. Larangan Data Test             : {OFFICIAL_RULES_SUMMARY['test_data_restriction']}")
print(f"3. Izin Pretrained Model          : {OFFICIAL_RULES_SUMMARY['pretrained_model_allowed']}")
print(f"4. Larangan Dataset Eksternal     : {OFFICIAL_RULES_SUMMARY['external_dataset_prohibited']}")
print(f"5. Larangan AutoML/LLM/GenAI      : {OFFICIAL_RULES_SUMMARY['automl_llm_genai_prohibited']}")
print(f"6. Status Pseudo-Labeling         : {OFFICIAL_RULES_SUMMARY['pseudo_labeling_status']}")
print(f"7. Kewajiban Single Notebook      : {OFFICIAL_RULES_SUMMARY['single_notebook_reproducible']}")
print(f"8. Batas Pengumpulan Submission   : {OFFICIAL_RULES_SUMMARY['submission_deadline']}")
print("=" * 75)

# Automated Compliance Assertions
assert CONFIG["max_labels_per_class"] <= 5, "Pelanggaran Regulasi: Kuota label per kelas melebihi 5!"
assert CONFIG["max_total_labels"] <= 40, "Pelanggaran Regulasi: Kuota total label manual melebihi 40!"
print("\n[KEPATUHAN REGULASI: TERVALIDASI OTOMATIS]")
print("Parameter konfigurasi telah memenuhi batasan ketat handbook lomba.")
print("Pipeline siap dieksekusi secara otomatis dari awal hingga akhir.")
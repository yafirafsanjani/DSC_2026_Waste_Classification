# tools/parts/part1_engine.py
"""
Manual Few-Shot Annotation Tool for DSC Gelar Rasa 2026.
Standalone utility to label exactly 5 images per class (40 images total)
from official unlabelled train set (data/competition/train/).
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASSES = sorted([
    "cardboard",
    "electronic",
    "foodwaste",
    "glass",
    "metal",
    "paper",
    "plastic",
    "textile"
])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

MAX_PER_CLASS = 5
MAX_TOTAL = 40

def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

class FewShotAnnotationEngine:
    """Core logic engine managing candidate images, quotas, persistence, and CSV export."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        train_dir: Optional[Path] = None,
        test_dir: Optional[Path] = None,
        manifest_dir: Optional[Path] = None
    ):
        self.project_root = (project_root or PROJECT_ROOT).resolve()
        self.train_dir = (train_dir or self.project_root / "data" / "competition" / "train").resolve()
        self.test_dir = (test_dir or self.project_root / "data" / "competition" / "test").resolve()
        self.manifest_dir = (manifest_dir or self.project_root / "data" / "manifests" / "fewshot_pseudolabel_b2_v1").resolve()
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = self.manifest_dir / "annotation_progress.json"
        self.output_csv = self.manifest_dir / "labeled_pool_40.csv"

        self.annotations: Dict[str, Dict] = {}
        self.annotation_history: List[str] = []
        self.skipped_ids: Set[str] = set()
        self.current_idx: int = 0

        self.test_hashes: Dict[str, str] = {}
        self.quarantined_files: List[Tuple[str, str, str]] = []
        self.candidate_files: List[Path] = []
        self._scan_and_filter_candidates()

        self.load_progress()

    def _scan_and_filter_candidates(self):
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Direktori data train resmi tidak ditemukan: {self.train_dir}")

        if self.test_dir.exists():
            test_files = [p for p in self.test_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            for p in test_files:
                h = compute_file_sha256(p)
                self.test_hashes[h] = p.name

        all_train = sorted([
            p for p in self.train_dir.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ], key=lambda p: p.name)

        self.candidate_files = []
        self.quarantined_files = []
        for p in all_train:
            h = compute_file_sha256(p)
            if h in self.test_hashes:
                self.quarantined_files.append((p.name, self.test_hashes[h], h))
            else:
                self.candidate_files.append(p)

    def get_class_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in CLASSES}
        for item in self.annotations.values():
            cname = item.get("class_name")
            if cname in counts:
                counts[cname] += 1
        return counts

    def get_total_count(self) -> int:
        return len(self.annotations)

    def is_complete(self) -> bool:
        counts = self.get_class_counts()
        return all(cnt == MAX_PER_CLASS for cnt in counts.values()) and len(self.annotations) == MAX_TOTAL

    def can_annotate_class(self, class_name: str) -> bool:
        if class_name not in CLASSES:
            return False
        if len(self.annotations) >= MAX_TOTAL:
            return False
        counts = self.get_class_counts()
        return counts[class_name] < MAX_PER_CLASS

    def get_current_candidate(self) -> Optional[Dict]:
        if self.is_complete():
            return None

        n = len(self.candidate_files)
        if n == 0:
            return None

        for i in range(n):
            idx = (self.current_idx + i) % n
            path = self.candidate_files[idx]
            img_id = path.stem
            if img_id not in self.annotations and img_id not in self.skipped_ids:
                self.current_idx = idx
                return {
                    "image_id": img_id,
                    "filename": path.name,
                    "path": path,
                    "relative_path": path.relative_to(self.project_root).as_posix(),
                    "sha256": compute_file_sha256(path),
                    "index": idx,
                    "total_candidates": n
                }

        return None

    def annotate_current(self, class_name: str) -> Tuple[bool, str]:
        if class_name not in CLASSES:
            return False, f"Kelas '{class_name}' tidak valid. Harus salah satu dari {CLASSES}."

        if not self.can_annotate_class(class_name):
            counts = self.get_class_counts()
            return False, f"Kuota kelas '{class_name}' sudah penuh ({counts[class_name]}/{MAX_PER_CLASS})!"

        cand = self.get_current_candidate()
        if not cand:
            return False, "Tidak ada citra kandidat yang tersedia untuk dilabeli."

        img_id = cand["image_id"]
        record = {
            "image_id": str(img_id),
            "class_name": class_name,
            "class_idx": CLASS_TO_IDX[class_name],
            "relative_path": cand["relative_path"],
            "sha256": cand["sha256"]
        }

        self.annotations[img_id] = record
        self.annotation_history.append(img_id)
        if img_id in self.skipped_ids:
            self.skipped_ids.remove(img_id)

        self.current_idx = (self.current_idx + 1) % len(self.candidate_files)
        self.save_progress()

        if self.is_complete():
            self.export_csv()
            return True, f"[LENGKAP] Citra '{img_id}' dilabeli '{class_name}'. Total 40 label terpenuhi! File CSV diekspor."

        counts = self.get_class_counts()
        return True, f"Citra '{img_id}' dilabeli '{class_name}' ({counts[class_name]}/{MAX_PER_CLASS}). Total: {len(self.annotations)}/{MAX_TOTAL}."

    def skip_current(self) -> Tuple[bool, str]:
        cand = self.get_current_candidate()
        if not cand:
            return False, "Tidak ada citra untuk dilewati."

        img_id = cand["image_id"]
        self.skipped_ids.add(img_id)
        self.current_idx = (self.current_idx + 1) % len(self.candidate_files)
        self.save_progress()
        return True, f"Citra '{img_id}' dilewati (skip)."

    def undo_last(self) -> Tuple[bool, str]:
        if not self.annotation_history:
            return False, "Tidak ada anotasi sebelumnya untuk dibatalkan."

        last_id = self.annotation_history.pop()
        removed_record = self.annotations.pop(last_id, None)
        if not removed_record:
            return False, f"ID '{last_id}' tidak ditemukan di memori anotasi."

        for idx, p in enumerate(self.candidate_files):
            if p.stem == last_id:
                self.current_idx = idx
                break

        self.save_progress()
        cname = removed_record.get("class_name", "")
        return True, f"Anotasi citra '{last_id}' ({cname}) berhasil dibatalkan."

    def reset_skips(self) -> Tuple[bool, str]:
        num_skips = len(self.skipped_ids)
        self.skipped_ids.clear()
        self.save_progress()
        return True, f"Daftar skip telah di-reset ({num_skips} citra dikembalikan ke antrean)."

    def reset_all_progress(self) -> Tuple[bool, str]:
        if self.progress_file.exists():
            backup_file = self.progress_file.with_suffix(".bak")
            self.progress_file.replace(backup_file)
        self.annotations.clear()
        self.annotation_history.clear()
        self.skipped_ids.clear()
        self.current_idx = 0
        return True, "Seluruh progres anotasi berhasil di-reset."

    def save_progress(self):
        data = {
            "annotations": self.annotations,
            "annotation_history": self.annotation_history,
            "skipped_ids": list(self.skipped_ids),
            "current_idx": self.current_idx,
            "counts": self.get_class_counts(),
            "total_annotated": len(self.annotations),
            "is_complete": self.is_complete()
        }
        temp_path = self.progress_file.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.progress_file)

    def load_progress(self):
        if not self.progress_file.exists():
            return
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.annotations = data.get("annotations", {})
            self.annotation_history = data.get("annotation_history", [])
            self.skipped_ids = set(data.get("skipped_ids", []))
            self.current_idx = data.get("current_idx", 0)
        except Exception as e:
            print(f"[PERINGATAN] Gagal memuat {self.progress_file.name}: {e}")

    def export_csv(self, target_path: Optional[Path] = None) -> Tuple[bool, str]:
        counts = self.get_class_counts()
        if not self.is_complete():
            missing = [f"{c}: {cnt}/{MAX_PER_CLASS}" for c, cnt in counts.items() if cnt < MAX_PER_CLASS]
            return False, f"Belum lengkap 40 label! Kekurangan: {', '.join(missing)}"

        records = list(self.annotations.values())
        df = pd.DataFrame(records)

        assert len(df) == MAX_TOTAL, f"Jumlah baris harus tepat {MAX_TOTAL}, didapat {len(df)}"
        assert set(df["class_name"]) == set(CLASSES), "Delapan kelas resmi harus hadir secara lengkap"
        assert not df["image_id"].duplicated().any(), "Tidak boleh ada duplikasi image_id"
        assert df["class_name"].value_counts().eq(MAX_PER_CLASS).all(), f"Setiap kelas harus tepat memiliki {MAX_PER_CLASS} sampel"

        for h in df["sha256"]:
            assert h not in self.test_hashes, f"Fatal: Citra dengan hash {h} bertabrakan dengan test set!"

        out_file = (target_path or self.output_csv).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df = df.sort_values("image_id").reset_index(drop=True)
        cols = ["image_id", "class_name", "class_idx", "relative_path", "sha256"]
        df[cols].to_csv(out_file, index=False)
        return True, f"File CSV Labeled Pool berhasil diekspor dan diverifikasi: {out_file}"


# tools/parts/part2_gui.py
class AnnotationGUI:
    """Tkinter Desktop GUI for intuitive, safe manual few-shot labeling."""

    def __init__(self, root, engine: FewShotAnnotationEngine):
        self.root = root
        self.engine = engine
        self.current_photo = None

        self.root.title("DSC Gelar Rasa 2026 — Manual Few-Shot Annotation Tool")
        self.root.geometry("1060x780")
        self.root.minsize(980, 700)

        self._configure_styles()
        self._build_layout()
        self._bind_keys()
        self._refresh_view()

    def _configure_styles(self):
        import tkinter.ttk as ttk
        style = ttk.Style(self.root)
        style.theme_use("clam")

        self.font_title = ("Helvetica", 14, "bold")
        self.font_subtitle = ("Helvetica", 9)
        self.font_header = ("Helvetica", 11, "bold")
        self.font_body = ("Helvetica", 10)
        self.font_button = ("Helvetica", 10, "bold")
        self.font_status = ("Consolas", 10)

    def _build_layout(self):
        import tkinter as tk
        import tkinter.ttk as ttk

        # 1. TOP HEADER FRAME
        header_frame = tk.Frame(self.root, bg="#1e293b", padx=16, pady=12)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="DSC Gelar Rasa 2026 — Few-Shot Manual Annotation Tool",
            font=self.font_title,
            fg="#f8fafc",
            bg="#1e293b"
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = tk.Label(
            header_frame,
            text="Regulasi: Maksimal 5 citra per kelas (Total 40 citra dari 1.920 data train resmi). Citra tabrakan test set dikarantina otomatis.",
            font=self.font_subtitle,
            fg="#94a3b8",
            bg="#1e293b"
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(2, 6))

        prog_subframe = tk.Frame(header_frame, bg="#1e293b")
        prog_subframe.pack(fill=tk.X, pady=(4, 0))

        self.progress_bar = ttk.Progressbar(prog_subframe, orient=tk.HORIZONTAL, maximum=MAX_TOTAL, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 12))

        self.progress_lbl = tk.Label(
            prog_subframe,
            text="Total Anotasi: 0 / 40 (0.0%)",
            font=self.font_button,
            fg="#38bdf8",
            bg="#1e293b"
        )
        self.progress_lbl.pack(side=tk.RIGHT)

        # 2. MAIN WORKSPACE
        workspace = tk.Frame(self.root, bg="#f1f5f9", padx=16, pady=12)
        workspace.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_pane = tk.Frame(workspace, bg="#ffffff", bd=1, relief=tk.SOLID, padx=12, pady=12)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.img_info_lbl = tk.Label(
            left_pane,
            text="Memuat citra kandidat...",
            font=self.font_header,
            fg="#0f172a",
            bg="#ffffff"
        )
        self.img_info_lbl.pack(anchor=tk.W, pady=(0, 8))

        self.img_canvas = tk.Label(left_pane, bg="#e2e8f0", text="Citra", width=55, height=22)
        self.img_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.meta_lbl = tk.Label(
            left_pane,
            text="Resolusi: - | Path: -",
            font=self.font_subtitle,
            fg="#475569",
            bg="#ffffff",
            anchor=tk.W,
            justify=tk.LEFT
        )
        self.meta_lbl.pack(fill=tk.X)

        right_pane = tk.Frame(workspace, bg="#f1f5f9", width=420)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_pane.pack_propagate(False)

        class_box = tk.LabelFrame(
            right_pane,
            text=" 1. PILIH KELAS SAMPAH (Maksimal 5 per kelas) ",
            font=self.font_header,
            fg="#1e293b",
            bg="#ffffff",
            padx=12,
            pady=10
        )
        class_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.class_buttons: Dict[str, tk.Button] = {}
        for idx, cname in enumerate(CLASSES, 1):
            btn = tk.Button(
                class_box,
                text=f"[{idx}] {cname} (0/5)",
                font=self.font_button,
                bg="#f8fafc",
                fg="#0f172a",
                activebackground="#3b82f6",
                activeforeground="#ffffff",
                relief=tk.RAISED,
                bd=2,
                anchor=tk.W,
                padx=10,
                pady=6,
                command=lambda c=cname: self._on_class_selected(c)
            )
            btn.pack(fill=tk.X, pady=3)
            self.class_buttons[cname] = btn

        ctrl_box = tk.LabelFrame(
            right_pane,
            text=" 2. NAVIGASI & KONTROL ",
            font=self.font_header,
            fg="#1e293b",
            bg="#ffffff",
            padx=12,
            pady=10
        )
        ctrl_box.pack(fill=tk.X, pady=(0, 0))

        btn_grid = tk.Frame(ctrl_box, bg="#ffffff")
        btn_grid.pack(fill=tk.X)

        self.skip_btn = tk.Button(
            btn_grid,
            text="[S] Lewati (Skip)",
            font=self.font_button,
            bg="#fef3c7",
            fg="#92400e",
            relief=tk.GROOVE,
            padx=8,
            pady=6,
            command=self._on_skip
        )
        self.skip_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.undo_btn = tk.Button(
            btn_grid,
            text="[U] Undo Terakhir",
            font=self.font_button,
            bg="#fee2e2",
            fg="#991b1b",
            relief=tk.GROOVE,
            padx=8,
            pady=6,
            command=self._on_undo
        )
        self.undo_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        btn_grid2 = tk.Frame(ctrl_box, bg="#ffffff")
        btn_grid2.pack(fill=tk.X, pady=(6, 0))

        self.reset_skip_btn = tk.Button(
            btn_grid2,
            text="[R] Reset Skips",
            font=self.font_subtitle,
            bg="#f1f5f9",
            fg="#475569",
            relief=tk.GROOVE,
            pady=4,
            command=self._on_reset_skips
        )
        self.reset_skip_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.export_btn = tk.Button(
            btn_grid2,
            text="[E] Ekspor CSV",
            font=self.font_button,
            bg="#dcfce7",
            fg="#166534",
            relief=tk.GROOVE,
            pady=4,
            command=self._on_export
        )
        self.export_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # 3. BOTTOM STATUS BAR
        status_frame = tk.Frame(self.root, bg="#0f172a", padx=16, pady=8)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_lbl = tk.Label(
            status_frame,
            text="[SIAP] Amati citra sampah di sebelah kiri, lalu tekan nomor kelas [1-8] atau tombol terkait.",
            font=self.font_status,
            fg="#38bdf8",
            bg="#0f172a",
            anchor=tk.W
        )
        self.status_lbl.pack(fill=tk.X)

    def _bind_keys(self):
        for idx, cname in enumerate(CLASSES, 1):
            self.root.bind(str(idx), lambda e, c=cname: self._on_class_selected(c))
            self.root.bind(f"<KP_{idx}>", lambda e, c=cname: self._on_class_selected(c))

        self.root.bind("s", lambda e: self._on_skip())
        self.root.bind("S", lambda e: self._on_skip())
        self.root.bind("<space>", lambda e: self._on_skip())
        self.root.bind("<Right>", lambda e: self._on_skip())

        self.root.bind("u", lambda e: self._on_undo())
        self.root.bind("U", lambda e: self._on_undo())
        self.root.bind("<Control-z>", lambda e: self._on_undo())

        self.root.bind("r", lambda e: self._on_reset_skips())
        self.root.bind("e", lambda e: self._on_export())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def _refresh_view(self):
        import tkinter as tk
        from PIL import ImageTk

        counts = self.engine.get_class_counts()
        total_annotated = self.engine.get_total_count()

        self.progress_bar["value"] = total_annotated
        pct = (total_annotated / MAX_TOTAL) * 100.0
        self.progress_lbl.config(text=f"Total Anotasi: {total_annotated} / {MAX_TOTAL} ({pct:.1f}%)")

        for idx, cname in enumerate(CLASSES, 1):
            cnt = counts[cname]
            btn = self.class_buttons[cname]
            if cnt >= MAX_PER_CLASS:
                btn.config(
                    text=f"[{idx}] {cname:11s} : 5 / 5  [PENUH ✓]",
                    state=tk.DISABLED,
                    bg="#dcfce7",
                    fg="#15803d",
                    relief=tk.FLAT
                )
            else:
                btn.config(
                    text=f"[{idx}] {cname:11s} : {cnt} / 5",
                    state=tk.NORMAL,
                    bg="#f8fafc",
                    fg="#0f172a",
                    relief=tk.RAISED
                )

        if self.engine.is_complete():
            self.img_info_lbl.config(
                text="🎉 SELURUH 40 ANOTASI TELAH LENGKAP!",
                fg="#16a34a"
            )
            self.img_canvas.config(
                image="",
                text="Semua 8 kelas telah memiliki tepat 5 sampel.\nBerkas 'labeled_pool_40.csv' berhasil diekspor!\n\nAnda dapat menutup aplikasi ini dan melanjutkan Run All pada notebook.",
                font=self.font_header,
                fg="#16a34a"
            )
            self.meta_lbl.config(text=f"Lokasi File: {self.engine.output_csv}")
            self.status_lbl.config(
                text="[SELESAI] Labeled pool 40 citra resmi telah siap dan terverifikasi untuk Section 08 notebook.",
                fg="#4ade80"
            )
            return

        cand = self.engine.get_current_candidate()
        if not cand:
            if len(self.engine.skipped_ids) > 0:
                self.img_info_lbl.config(text="Semua citra yang tersisa saat ini ada di daftar Skip.", fg="#d97706")
                self.img_canvas.config(
                    image="",
                    text=f"Seluruh citra belum terlabeli telah Anda lewati ({len(self.engine.skipped_ids)} citra).\n\nTekan tombol '[R] Reset Skips' di sebelah kanan\nuntuk mengulang kembali citra yang dilewati.",
                    font=self.font_header,
                    fg="#d97706"
                )
            else:
                self.img_info_lbl.config(text="Tidak ada citra kandidat yang tersedia.", fg="#dc2626")
                self.img_canvas.config(image="", text="Tidak ada citra train yang ditemukan.")
            return

        img_path = cand["path"]
        try:
            pil_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            max_w, max_h = 480, 420
            ratio = min(max_w / orig_w, max_h / orig_h)
            new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            self.current_photo = ImageTk.PhotoImage(resized)
            self.img_canvas.config(image=self.current_photo, text="")

            idx_display = cand["index"] + 1
            tot_display = cand["total_candidates"]
            self.img_info_lbl.config(
                text=f"Citra ID: {cand['image_id']}  ({cand['filename']})  —  [Kandidat {idx_display}/{tot_display}]",
                fg="#0f172a"
            )
            self.meta_lbl.config(
                text=f"Resolusi Asli: {orig_w}x{orig_h} px  |  Hash SHA-256: {cand['sha256'][:16]}... (Bebas tabrakan test set)"
            )
        except Exception as e:
            self.img_canvas.config(image="", text=f"Gagal memuat citra:\n{e}")

    def _on_class_selected(self, class_name: str):
        if not self.engine.can_annotate_class(class_name):
            import tkinter.messagebox as msgbox
            msgbox.showwarning(
                "Kuota Penuh",
                f"Kelas '{class_name}' sudah memiliki 5 sampel (kuota maksimal)!\nSilakan pilih kelas lain yang belum penuh."
            )
            return

        success, msg = self.engine.annotate_current(class_name)
        self.status_lbl.config(text=f"[{'SUKSES' if success else 'GAGAL'}] {msg}", fg="#38bdf8" if success else "#f87171")
        self._refresh_view()

        if self.engine.is_complete():
            import tkinter.messagebox as msgbox
            msgbox.showinfo(
                "Anotasi Lengkap!",
                "Selamat! Seluruh 8 kelas telah memiliki tepat 5 sampel (total 40 anotasi sah).\n\n"
                f"Berkas Labeled Pool berhasil diekspor dan diverifikasi pada:\n"
                f"  {self.engine.output_csv}\n\n"
                "Pipeline sekarang siap dieksekusi secara otomatis melalui 'Kernel Restart & Run All' di notebook 08."
            )

    def _on_skip(self):
        success, msg = self.engine.skip_current()
        self.status_lbl.config(text=f"[SKIP] {msg}", fg="#facc15")
        self._refresh_view()

    def _on_undo(self):
        success, msg = self.engine.undo_last()
        self.status_lbl.config(text=f"[{'UNDO' if success else 'GAGAL'}] {msg}", fg="#fb7185")
        self._refresh_view()

    def _on_reset_skips(self):
        success, msg = self.engine.reset_skips()
        self.status_lbl.config(text=f"[RESET] {msg}", fg="#38bdf8")
        self._refresh_view()

    def _on_export(self):
        import tkinter.messagebox as msgbox
        success, msg = self.engine.export_csv()
        if success:
            msgbox.showinfo("Ekspor Berhasil", msg)
            self.status_lbl.config(text=f"[EKSPOR] {msg}", fg="#4ade80")
        else:
            msgbox.showwarning("Ekspor Belum Lengkap", msg)
            self.status_lbl.config(text=f"[EKSPOR DITOLAK] {msg}", fg="#f87171")


# tools/parts/part3_cli.py
class AnnotationCLI:
    """Interactive command-line interface for terminal environments."""

    def __init__(self, engine: FewShotAnnotationEngine):
        self.engine = engine

    def run(self):
        print("\n" + "=" * 75)
        print("DSC GELAR RASA 2026 — MANUAL FEW-SHOT ANNOTATION TOOL (CLI MODE)")
        print("Regulasi: Maksimal 5 citra per kelas (Total 40 citra dari data train resmi).")
        print("=" * 75)

        while True:
            if self.engine.is_complete():
                print("\n🎉 SELURUH 40 ANOTASI TELAH LENGKAP (5 PER KELAS x 8 KELAS)!")
                success, msg = self.engine.export_csv()
                print(msg)
                print(f"Berkas tersimpan di: {self.engine.output_csv}")
                break

            cand = self.engine.get_current_candidate()
            if not cand:
                if len(self.engine.skipped_ids) > 0:
                    print(f"\n[PERINGATAN] Seluruh citra tersisa ada di daftar skip ({len(self.engine.skipped_ids)} citra).")
                    resp = input("Reset daftar skip untuk mengulang kembali? (y/n): ").strip().lower()
                    if resp == "y":
                        self.engine.reset_skips()
                        continue
                print("\nTidak ada citra kandidat yang tersedia.")
                break

            counts = self.engine.get_class_counts()
            total_annotated = self.engine.get_total_count()

            print("\n" + "-" * 75)
            print(f"CITRA KANDIDAT: ID = {cand['image_id']} ({cand['filename']}) | Total Progres: {total_annotated}/{MAX_TOTAL}")
            print(f"Path: {cand['relative_path']} | SHA-256: {cand['sha256'][:16]}...")
            print("-" * 75)
            print("PILIHAN KELAS:")
            for idx, cname in enumerate(CLASSES, 1):
                cnt = counts[cname]
                status = "[PENUH]" if cnt >= MAX_PER_CLASS else f"({cnt}/5)"
                print(f"  [{idx}] {cname:12s} {status}")

            print("\nKONTROL: [s] Lewati (Skip) | [u] Undo | [r] Reset Skips | [e] Ekspor | [q] Keluar")

            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(cand["path"]))
                else:
                    Image.open(cand["path"]).show()
            except Exception:
                pass

            choice = input("\nMasukkan pilihan (1-8, s, u, r, e, q): ").strip().lower()

            if choice in [str(i) for i in range(1, 9)]:
                idx = int(choice) - 1
                cname = CLASSES[idx]
                success, msg = self.engine.annotate_current(cname)
                print(f">> {msg}")
            elif choice == "s":
                success, msg = self.engine.skip_current()
                print(f">> {msg}")
            elif choice == "u":
                success, msg = self.engine.undo_last()
                print(f">> {msg}")
            elif choice == "r":
                success, msg = self.engine.reset_skips()
                print(f">> {msg}")
            elif choice == "e":
                success, msg = self.engine.export_csv()
                print(f">> {msg}")
            elif choice == "q":
                print("\nKeluar dari alat anotasi. Progres Anda telah disimpan.")
                break
            else:
                print("Pilihan tidak valid!")


def main():
    parser = argparse.ArgumentParser(description="Manual Few-Shot Annotation Tool for DSC 2026")
    parser.add_argument("--cli", action="store_true", help="Jalankan dalam mode terminal CLI interaktif")
    parser.add_argument("--info", action="store_true", help="Tampilkan ringkasan status dan kuota anotasi saat ini")
    parser.add_argument("--export", action="store_true", help="Ekspor berkas labeled_pool_40.csv jika sudah lengkap")
    parser.add_argument("--reset", action="store_true", help="Reset seluruh riwayat progres anotasi")
    args = parser.parse_args()

    engine = FewShotAnnotationEngine()

    if args.info:
        counts = engine.get_class_counts()
        print("\n" + "=" * 65)
        print("STATUS MANUAL FEW-SHOT ANNOTATION:")
        print(f"  Total Anotasi   : {engine.get_total_count()} / {MAX_TOTAL} citra")
        print(f"  Status Lengkap  : {'LENGKAP (Siap Run All)' if engine.is_complete() else 'BELUM LENGKAP'}")
        print("  Distribusi Kuota per Kelas (Maks 5/kelas):")
        for cname in CLASSES:
            cnt = counts[cname]
            st = "PENUH ✓" if cnt >= MAX_PER_CLASS else f"{cnt}/{MAX_PER_CLASS}"
            print(f"    - {cname:12s}: {st}")
        print(f"  Citra di-skip   : {len(engine.skipped_ids)} citra")
        print(f"  File Progres    : {engine.progress_file}")
        print(f"  File Labeled CSV: {engine.output_csv} ({'ADA' if engine.output_csv.exists() else 'BELUM ADA'})")
        print("=" * 65)
        return

    if args.reset:
        confirm = input("Apakah Anda yakin ingin me-reset seluruh progres anotasi? (y/n): ").strip().lower()
        if confirm == "y":
            _, msg = engine.reset_all_progress()
            print(msg)
        return

    if args.export:
        success, msg = engine.export_csv()
        print(msg)
        return

    if args.cli:
        cli = AnnotationCLI(engine)
        cli.run()
        return

    try:
        import tkinter as tk
        root = tk.Tk()
        app = AnnotationGUI(root, engine)
        root.mainloop()
    except Exception as e:
        print(f"[PERINGATAN] Gagal membuka antarmuka GUI ({e}). Beralih otomatis ke mode CLI...")
        cli = AnnotationCLI(engine)
        cli.run()


if __name__ == "__main__":
    main()

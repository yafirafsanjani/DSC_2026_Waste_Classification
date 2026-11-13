# ==============================================================================
# SECTION 28: PSEUDO-LABEL GENERATION
# ==============================================================================
RAW_PSEUDO_PATH = MANIFEST_DIR / "pseudo_labels.csv"

def generate_raw_pseudo_labels(
    model: nn.Module,
    unlabeled_df: pd.DataFrame,
    output_path: Path,
    batch_size: int = 16,
    device: torch.device = DEVICE
) -> pd.DataFrame:
    dataset = UnlabeledDataset(unlabeled_df, PROJECT_ROOT, transform=eval_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    records = []
    print(f"Menghasilkan pseudo-label pada {len(dataset)} citra unlabeled...")
    
    model.eval()
    with torch.inference_mode():
        for images, image_ids in tqdm(loader, desc="Pseudo-Labeling Inference"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            for img_id, pred_idx, conf in zip(image_ids, preds.cpu().numpy(), confidences.cpu().numpy()):
                records.append({
                    "image_id": str(img_id),
                    "predicted_idx": int(pred_idx),
                    "predicted_class": IDX_TO_CLASS[int(pred_idx)],
                    "confidence": float(conf),
                    "confidence_score": float(conf),
                    "teacher_checkpoint": TEACHER_BEST_CKPT.name if TEACHER_BEST_CKPT.exists() else "teacher_best"
                })
                
    df_pred = pd.DataFrame(records)
    merged = pd.merge(unlabeled_df[["image_id", "relative_path", "sha256"]], df_pred, on="image_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged

if RAW_PSEUDO_PATH.exists():
    raw_pseudo_df = pd.read_csv(RAW_PSEUDO_PATH, dtype={"image_id": str})
    print(f"[LOADED] Pseudo-labels dimuat dari {RAW_PSEUDO_PATH} ({len(raw_pseudo_df)} baris).")
elif inference_teacher is not None and len(unlabeled_pool_df) > 0:
    print(f"[RUN-ALL] Menghasilkan pseudo-label pada {len(unlabeled_pool_df)} citra unlabeled...")
    raw_pseudo_df = generate_raw_pseudo_labels(inference_teacher, unlabeled_pool_df, output_path=RAW_PSEUDO_PATH, batch_size=16)
    print(f"[COMPLETED] Pseudo-labels berhasil disimpan ke {RAW_PSEUDO_PATH}")
else:
    print("[STATUS] Prasyarat inference_teacher atau unlabeled pool belum siap.")
    raw_pseudo_df = pd.DataFrame()
# ==============================================================================
# SECTION 51: COMPETITION TEST INFERENCE
# ==============================================================================
def predict_competition_test(
    ckpt_path: Path,
    test_dir: Path,
    sample_sub_path: Path,
    device: torch.device
) -> Dict[str, str]:
    sample_df = pd.read_csv(sample_sub_path, dtype={"image_id": str})
    expected_ids = sample_df["image_id"].tolist()
    
    model = build_teacher_model(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Resolusi ekstensi file (.jpg, .jpeg, .png) secara dinamis
    test_file_map = {p.stem: p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    test_files = [test_file_map[str(img_id)] for img_id in expected_ids]
    test_ds = CompetitionTestDataset(test_files, transform=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    predictions = {}
    print(f"Menjalankan inferensi pada {len(test_ds)} citra test...")
    with torch.inference_mode():
        for images, img_ids in tqdm(test_loader, desc="Test Inference"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            for img_id, pred_idx in zip(img_ids, preds):
                predictions[str(img_id)] = IDX_TO_CLASS[int(pred_idx)]
                
    return predictions

if SELECTED_CKPT_PATH is not None and SELECTED_CKPT_PATH.exists():
    print(f"[RUN-ALL] Menjalankan inferensi test kompetisi menggunakan {SELECTED_MODEL_TYPE}...")
    test_predictions = predict_competition_test(SELECTED_CKPT_PATH, DATA_TEST_DIR, SAMPLE_SUBMISSION_PATH, DEVICE)
    print(f"[COMPLETED] Inferensi selesai. {len(test_predictions)} prediksi berhasil dibuat.")
else:
    print("[STATUS] Checkpoint model untuk inferensi belum tersedia.")
    test_predictions = {}
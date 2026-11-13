# ==============================================================================
# SECTION 42: TEACHER VS STUDENT COMPARISON
# ==============================================================================
def compare_teacher_and_student():
    records = []
    
    # Data Teacher
    if TEACHER_BEST_CKPT.exists():
        t_payload = torch.load(TEACHER_BEST_CKPT, map_location="cpu")
        records.append({
            "Model": "Final Teacher",
            "Backbone": "EfficientNet-B2",
            "Supervised Samples": 40,
            "Pseudo Samples": 0,
            "Final Loss": f"{t_payload.get('final_loss', float('nan')):.4f}",
            "Training Acc": f"{t_payload.get('final_acc', float('nan'))*100:.2f}%",
            "Role": "Initial Pseudo-Label Generator"
        })
        
    # Data Student
    if STUDENT_BEST_CKPT.exists():
        s_payload = torch.load(STUDENT_BEST_CKPT, map_location="cpu")
        num_pseudo = len(accepted_pseudo_df) if "accepted_pseudo_df" in globals() else 0
        records.append({
            "Model": "Student",
            "Backbone": "EfficientNet-B2",
            "Supervised Samples": 40,
            "Pseudo Samples": num_pseudo,
            "Final Loss": f"{s_payload.get('final_loss', float('nan')):.4f}",
            "Training Acc": f"{s_payload.get('final_acc', float('nan'))*100:.2f}%",
            "Role": "Final Classifier with Distillation"
        })
        
    if records:
        df_comp = pd.DataFrame(records)
        print("TABEL PERBANDINGAN TEACHER VS STUDENT:")
        print(df_comp.to_string(index=False))
    else:
        print("[INFO] Model Teacher atau Student belum selesai dilatih.")

compare_teacher_and_student()
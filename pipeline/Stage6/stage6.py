import os
from phase1 import Phase1
from phase2 import Phase2

# Config
PREV_STAGE = "../Stage5"
DATA_DIR = "training dataset"
PHASE1_OUT = "phase1-results"
PHASE2_OUT = "phase2-results"

# Extracted from notebook analysis for best performance
SEED = 0
THRESHOLD = 0.614


def run_stage6():
    print("=======================================================")
    print("      [STAGE 6] PHASE 1: CLINICAL BRAIN (TABULAR)      ")
    print("=======================================================")

    # Initialize Phase 1
    p1 = Phase1(
        prev_stage=PREV_STAGE,
        data_dir=DATA_DIR,
        train_file="train_split.csv",
        val_file="val_split.csv",
        test_file="test_split.csv",
        feature_groups_file="engineered_feature_groups.json",
        seed=SEED,
        threshold=THRESHOLD
    )

    print("[STAGE 6] Training Part 1 (Clinical Only)...")
    report_p1 = p1.train_phase_part(part=1, output_dir=PHASE1_OUT)
    print(f" -> Test AUC: {report_p1['test_auc']:.4f} | Test Acc: {report_p1['test_accuracy']:.4f}")

    print("\n[STAGE 6] Training Part 2 (Clinical + Labs)...")
    # By omitting k_features, phase1.py automatically applies the optimized k=30
    report_p2 = p1.train_phase_part(part=2, output_dir=PHASE1_OUT)
    print(f" -> Test AUC: {report_p2['test_auc']:.4f} | Test Acc: {report_p2['test_accuracy']:.4f}")

    print(f"\n[STAGE 6] Phase 1 SUCCESS. Embeddings and metrics saved to {PHASE1_OUT}")

    print("\n=======================================================")
    print("      [STAGE 6] PHASE 2: IMAGE BRAIN (ULTRASOUND)      ")
    print("=======================================================")

    # Initialize Phase 2 (Image loader with smart padding & dual-screen split)
    p2 = Phase2(
        prev_stage=PREV_STAGE,
        data_dir=DATA_DIR,
        train_file="train_split.csv",
        val_file="val_split.csv",
        test_file="test_split.csv",
        seed=SEED
    )

    # Run Image CNN
    p2.train_phase2(output_dir=PHASE2_OUT)

    print("\n[STAGE 6] COMPLETE! Both Clinical and Image Embeddings are safely stored.")


if __name__ == "__main__":
    run_stage6()
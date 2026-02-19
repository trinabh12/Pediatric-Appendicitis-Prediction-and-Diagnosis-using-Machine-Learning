import os
from phase1 import Phase1

# Configurations
PREV_STAGE = "../Stage5"
DATA_DIR = "training dataset/tabular"
TRAIN_FILE = "train_split.csv"
VAL_FILE = "val_split.csv"
TEST_FILE = "test_split.csv"
FEATURE_REPORT = "engineered_feature_groups.json"

OUTPUT_ROOT = "models/phase1"
SEED = 0
THRESHOLD = 0.62


def run_stage6_phase1():
    print(f"[STAGE 6] Starting Phase 1 Model Training...")

    # Initialize Phase 1 Trainer
    p1 = Phase1(PREV_STAGE, DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE, FEATURE_REPORT, seed=SEED)

    # Execute Part 1: Demographics + Clinical
    print("[STAGE 6] Training Part 1 (Clinical Only)...")
    report1 = p1.train_phase_part(part=1, threshold=THRESHOLD, output_dir=OUTPUT_ROOT)
    print(f"[STAGE 6] Part 1 AUC: {report1['auc']:.4f}")

    # Execute Part 2: Demographics + Clinical + Labs
    print("[STAGE 6] Training Part 2 (Clinical + Labs)...")
    report2 = p1.train_phase_part(part=2, threshold=THRESHOLD, output_dir=OUTPUT_ROOT)
    print(f"[STAGE 6] Part 2 AUC: {report2['auc']:.4f}")

    print(f"[STAGE 6] Phase 1 Training complete. Assets saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    run_stage6_phase1()
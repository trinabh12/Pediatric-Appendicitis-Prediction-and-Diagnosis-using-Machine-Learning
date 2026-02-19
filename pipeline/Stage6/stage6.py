from phase1 import Phase1

# Config
PREV_STAGE = "../Stage5"
DATA_DIR = "training dataset"
OUTPUT_ROOT = "phase1-results"

# Extracted from notebook analysis for best performance
SEED = 0
THRESHOLD = 0.614

def run_stage6_phase1():
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
    report_p1 = p1.train_phase_part(part=1, output_dir=OUTPUT_ROOT)
    print(f" -> Test AUC: {report_p1['test_auc']:.4f} | Test Acc: {report_p1['test_accuracy']:.4f}")

    print("\n[STAGE 6] Training Part 2 (Clinical + Labs)...")
    report_p2 = p1.train_phase_part(part=2, output_dir=OUTPUT_ROOT)
    print(f" -> Test AUC: {report_p2['test_auc']:.4f} | Test Acc: {report_p2['test_accuracy']:.4f}")

    print(f"\n[STAGE 6] SUCCESS. Embeddings and comprehensive metrics saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    run_stage6_phase1()
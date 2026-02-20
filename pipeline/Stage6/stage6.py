import os
import shutil
from phase1 import Phase1
from phase2 import Phase2
from phase3 import Phase3

# Config
PREV_STAGE = "../Stage5"
DATA_DIR = "training dataset"
PHASE1_OUT = "phase1-results"
PHASE2_OUT = "phase2-results"
FINAL_OUT = "Final_Output"

SEED = 0
THRESHOLD = 0.614


def run_stage6():
    print("=======================================================")
    print("      [STAGE 6] PHASE 1: CLINICAL BRAIN (TABULAR)      ")
    print("=======================================================")
    p1 = Phase1(PREV_STAGE, DATA_DIR, "train_split.csv", "val_split.csv", "test_split.csv",
                "engineered_feature_groups.json", SEED, THRESHOLD)
    # Part 1 (Clinical)
    p1.train_phase_part(part=1, output_dir=PHASE1_OUT)
    # Part 2 (Clinical + Labs) - This generates the embeddings we actually use
    p1.train_phase_part(part=2, output_dir=PHASE1_OUT)

    print("\n=======================================================")
    print("      [STAGE 6] PHASE 2: IMAGE BRAIN (ULTRASOUND)      ")
    print("=======================================================")
    p2 = Phase2(PREV_STAGE, DATA_DIR, "train_split.csv", "val_split.csv", "test_split.csv", SEED)
    p2.train_phase2(output_dir=PHASE2_OUT)

    print("\n=======================================================")
    print("    [STAGE 6] PHASE 3: MULTI-TASK FUSION NETWORK       ")
    print("=======================================================")

    # Clean/Create Final Output Directory
    if os.path.exists(FINAL_OUT):
        shutil.rmtree(FINAL_OUT)
    os.makedirs(FINAL_OUT)

    # Initialize Phase 3
    p3 = Phase3(
        tabular_dir=os.path.join(PREV_STAGE, DATA_DIR, "tabular"),
        phase1_dir=os.path.join(PHASE1_OUT, "part_2"),  # Use Part 2 embeddings
        phase2_dir=PHASE2_OUT,
        seed=SEED
    )

    # Train and Save to FINAL_OUT
    p3.train_fusion(output_dir=FINAL_OUT)

    print("\n=======================================================")
    print(f"       PIPELINE COMPLETE.                          ")
    print(f"       FINAL MODEL SAVED IN: {FINAL_OUT}           ")
    print("=======================================================")


if __name__ == "__main__":
    run_stage6()
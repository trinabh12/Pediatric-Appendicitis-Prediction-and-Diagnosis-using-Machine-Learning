import os
import sys
from prefect import task, flow, get_run_logger


# Resolve the absolute path to the project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Dynamically add every Stage folder to sys.path
for i in range(1, 7):
    stage_dir = os.path.join(ROOT_DIR, "pipeline", f"Stage{i}")
    if os.path.exists(stage_dir) and stage_dir not in sys.path:
        sys.path.append(stage_dir)

# ==========================================
#  MODULAR IMPORTS
# ==========================================
try:
    from pipeline.Stage1.stage1 import run_stage1
    from pipeline.Stage2.stage2 import run_stage2
    from pipeline.Stage3.stage3 import run_stage3
    from pipeline.Stage4.stage4 import run_stage4
    from pipeline.Stage5.stage5 import run_stage5
    from pipeline.Stage6.stage6 import run_stage6
except ImportError as e:
    print(f"\n[CRITICAL ARCHITECTURE ERROR] Module Import Failed: {e}")
    sys.exit(1)


# ==========================================
# PREFECT TASK WRAPPERS
# ==========================================

@task(name="Stage 1: Data Ingestion", retries=0)
def execute_stage1():
    logger = get_run_logger()
    logger.info("🚀 Triggering Stage 1: Data Acquisition...")

    original_dir = os.getcwd()
    os.chdir(os.path.join(ROOT_DIR, "pipeline", "Stage1"))
    try:
        run_stage1()
    finally:
        os.chdir(original_dir)  # Always return to the root directory

    return True


@task(name="Stage 2: Clinical Validation")
def execute_stage2(prev_signal):
    logger = get_run_logger()
    logger.info("Triggering Stage 2: Data Validation...")

    original_dir = os.getcwd()
    os.chdir(os.path.join(ROOT_DIR, "pipeline", "Stage2"))
    try:
        run_stage2()
    finally:
        os.chdir(original_dir)

    return True


@task(name="Stage 3: Data Preparation")
def execute_stage3(prev_signal):
    logger = get_run_logger()
    logger.info("Triggering Stage 3: Preparation & Splitting...")

    original_dir = os.getcwd()
    os.chdir(os.path.join(ROOT_DIR, "pipeline", "Stage3"))
    try:
        run_stage3()
    finally:
        os.chdir(original_dir)

    return True


@task(name="Stage 4: Feature Engineering")
def execute_stage4(prev_signal):
    logger = get_run_logger()
    logger.info("🧬 Triggering Stage 4: Feature Engineering...")

    original_dir = os.getcwd()
    os.chdir(os.path.join(ROOT_DIR, "pipeline", "Stage4"))
    try:
        run_stage4()
    finally:
        os.chdir(original_dir)

    return True


@task(name="Stage 5: Data Versioning")
def execute_stage5(prev_signal):
    logger = get_run_logger()
    logger.info("Triggering Stage 5: Versioning & Lineage...")

    original_dir = os.getcwd()
    os.chdir(os.path.join(ROOT_DIR, "pipeline", "Stage5"))
    try:
        run_stage5()
    finally:
        os.chdir(original_dir)

    return True


@task(name="Stage 6: Neural Fusion Export")
def execute_stage6(prev_signal):
    logger = get_run_logger()
    logger.info("Triggering Stage 6: Fusion & Operationalization...")

    original_dir = os.getcwd()
    os.chdir(os.path.join(ROOT_DIR, "pipeline", "Stage6"))
    try:
        run_stage6()
    finally:
        os.chdir(original_dir)

    return True


# ==========================================
# THE MAIN WORKFLOW
# ==========================================

@flow(name="Pediatric-Appendicitis-CDSS-Pipeline")
def run_full_pipeline():
    logger = get_run_logger()
    logger.info("🎬 INITIALIZING END-TO-END PIPELINE ORCHESTRATION")

    # The Dependency Chain (Directed Acyclic Graph)
    s1 = execute_stage1()
    s2 = execute_stage2(s1)
    s3 = execute_stage3(s2)
    s4 = execute_stage4(s3)
    s5 = execute_stage5(s4)
    execute_stage6(s5)

    logger.info("ALL PIPELINE STAGES EXECUTED SUCCESSFULLY. READY FOR API.")


if __name__ == "__main__":
    run_full_pipeline()
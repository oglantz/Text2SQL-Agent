"""Central config: paths, model names, endpoints, hyperparameters.

Placeholder scaffold — fill in as phases are implemented. Keep all tunables here
so the frozen harness code (agent/tools/eval) never hard-codes a model or path.
"""
import os
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SPIDER_DIR = DATA_DIR / "spider" / "spider_data"
TRAJECTORIES_DIR = DATA_DIR / "trajectories"
SFT_DIR = DATA_DIR / "sft"
RESULTS_DIR = ROOT / "results"
EVAL_RUNS = RESULTS_DIR / "eval_runs.jsonl"

# Spider files
DEV_JSON = SPIDER_DIR / "dev.json"
TRAIN_JSON = SPIDER_DIR / "train_spider.json"
TABLES_JSON = SPIDER_DIR / "tables.json"
DATABASE_DIR = SPIDER_DIR / "database"  # <db_id>/<db_id>.sqlite

# --- Official Spider test-suite eval (used by `eval.py --official`) ---
# Clone https://github.com/taoyds/test-suite-sql-eval and point SPIDER_TEST_SUITE_DIR
# at it; TEST_SUITE_DB_DIR is the perturbed multi-instance databases (separate
# download). Either can be overridden via the matching environment variable.
SPIDER_TEST_SUITE_DIR = Path(
    os.environ.get("SPIDER_TEST_SUITE_DIR", ROOT / "third_party" / "test-suite-sql-eval")
)
TEST_SUITE_DB_DIR = Path(
    os.environ.get("SPIDER_TEST_SUITE_DB", SPIDER_DIR / "test_suite_database")
)

# --- Models / endpoints ---
# Teacher = hosted OpenAI-compatible API. Student = vLLM serving an open model.
TEACHER_BASE_URL = "https://api.deepseek.com"
TEACHER_MODEL = "deepseek-chat"
STUDENT_BASE_URL = "http://localhost:8000/v1"
STUDENT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# --- Agent loop ---
MAX_STEPS = 8
EVAL_TEMPERATURE = 0.0      # deterministic for teacher + eval
ROLLOUT_TEMPERATURE = 0.8   # diverse student rollouts for rejection sampling
QUERY_TIMEOUT_S = 5
MAX_OBSERVATION_ROWS = 30   # truncate tool observations

# --- QLoRA / SFT (set on Colab from your length distribution) ---
MAX_SEQ_LENGTH = 4096
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

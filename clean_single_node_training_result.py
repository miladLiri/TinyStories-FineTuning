import shutil
from pathlib import Path

TRAINING_DIR = Path("./single_node_finetuning")


def main():
    if TRAINING_DIR.exists():
        shutil.rmtree(TRAINING_DIR)
        print(f"Deleted '{TRAINING_DIR}' and all training artifacts.")
    else:
        print(f"'{TRAINING_DIR}' does not exist.")


if __name__ == "__main__":
    main()

import shutil
from pathlib import Path

TARGET_DIRS = [
    Path("./artifacts"),
    Path("./fine-tuning-dataset"),
]


def main():
    for target in TARGET_DIRS:
        if target.exists():
            shutil.rmtree(target)
            print(f"Deleted '{target}' and its contents.")
        else:
            print(f"'{target}' does not exist.")


if __name__ == "__main__":
    main()

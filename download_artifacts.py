import json
import os
import sys
import tarfile
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# UTF-8 console output guard for Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Hardcoded configuration
MODEL_ID = "roneneldan/TinyStories-1M"
DATASET_ID = "roneneldan/TinyStories"
DATASET_ARCHIVE = "TinyStories_all_data.tar.gz"

ARTIFACTS_DIR = Path("./artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"
DATASET_DIR = ARTIFACTS_DIR / "dataset"
FINETUNING_DIR = Path("./fine-tuning-dataset")

# Fine-tuning dataset split sizes
PREPARED_TRAIN_SAMPLES = 50000
PREPARED_VAL_SAMPLES = 2500


def download_pretrained_model():
    print(f"Downloading pretrained model and tokenizer for '{MODEL_ID}'...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    # Save pretrained weights and configs locally
    print(f"Saving pretrained model to '{MODEL_DIR}'...")
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print("Pretrained model and tokenizer download complete.\n")


def download_native_json_dataset():
    first_json = DATASET_DIR / "data00.json"
    if first_json.exists():
        print(f"Native JSON dataset already present in '{DATASET_DIR}'. Skipping re-download.\n")
        return

    print(f"Downloading native JSON dataset archive '{DATASET_ARCHIVE}' from '{DATASET_ID}'...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    archive_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=DATASET_ARCHIVE,
        repo_type="dataset",
        local_dir=str(DATASET_DIR),
    )
    print(f"Archive downloaded to '{archive_path}'.")

    # Extract native .json files from archive
    print(f"Extracting native JSON files into '{DATASET_DIR}'...")
    json_count = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if member.name.endswith(".json"):
                # Extract file directly into DATASET_DIR (stripping leading paths if any)
                member.name = Path(member.name).name
                tar.extract(member, path=str(DATASET_DIR))
                json_count += 1
                if json_count % 10 == 0:
                    print(f"  Extracted {json_count} native JSON files...")

    # Remove archive after extraction to save disk space
    try:
        os.remove(archive_path)
    except OSError:
        pass

    print(f"Native JSON dataset download complete. Extracted {json_count} JSON files.\n")


def create_finetuning_dataset():
    print("=" * 65)
    print(f"[*] Preparing instruction fine-tuning dataset under '{FINETUNING_DIR}'...")
    print("=" * 65)
    FINETUNING_DIR.mkdir(parents=True, exist_ok=True)

    source_file = DATASET_DIR / "data00.json"
    if not source_file.exists():
        print(f"[!] Source dataset file '{source_file}' not found. Cannot prepare fine-tuning dataset.")
        return

    print(f"[*] Loading raw stories from '{source_file}'...")
    with open(source_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    prepared_items = []
    for item in raw_items:
        story = item.get("story", "").strip()
        if not story:
            continue

        instruction = item.get("instruction", {})
        words = instruction.get("words", [])

        if words:
            prompt = f"Instruction: Write a short story using the words: {', '.join(words)}.\nStory:\n"
        else:
            prompt = "Instruction: Write a short story.\nStory:\n"

        prepared_items.append({
            "prompt": prompt,
            "story": story,
            "target_words": words,
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
        })

    train_data = prepared_items[:PREPARED_TRAIN_SAMPLES]
    val_data = prepared_items[PREPARED_TRAIN_SAMPLES : PREPARED_TRAIN_SAMPLES + PREPARED_VAL_SAMPLES]

    train_path = FINETUNING_DIR / "train.json"
    val_path = FINETUNING_DIR / "val.json"

    print(f"[*] Saving {len(train_data):,} training samples to '{train_path}'...")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    print(f"[*] Saving {len(val_data):,} validation samples to '{val_path}'...")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

    print(f"[+] Fine-tuning dataset created successfully in '{FINETUNING_DIR}':")
    print(f"    - train.json: {len(train_data):,} samples ({train_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"    - val.json:   {len(val_data):,} samples ({val_path.stat().st_size / (1024*1024):.1f} MB)\n")


def main():
    download_pretrained_model()
    download_native_json_dataset()
    create_finetuning_dataset()
    print("All requested artifacts and prepared datasets are ready.")


if __name__ == "__main__":
    main()

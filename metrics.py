import torch
import evaluate
import os
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


MODEL_DIR = "D:/College/M.Tech/INLP/SARVAM"
DATA_PATH = "data/test.jsonl"
OUTS_DIR = "out"
OUTPUT_FILE = os.path.join(OUTS_DIR, "combined_metrics_results.txt")


os.makedirs(OUTS_DIR, exist_ok=True)

# Load tokenizer and model
logger.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    logger.warning("Tokenizer has no pad_token. Assigning eos_token as pad_token.")
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, device_map="auto"
)

# Load test dataset
logger.info("Loading test dataset...")
test_dataset = load_dataset("json", data_files=DATA_PATH)["train"]


# Tokenization function
def tokenize_function(example):
    input_text = example["Instructions"] + " " + example["Input"]
    input_encoding = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",
    )
    target_encoding = tokenizer(
        example["Output"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",
    )

    return {
        "input_ids": input_encoding.input_ids[0],
        "attention_mask": input_encoding.attention_mask[0],
        "labels": target_encoding.input_ids[0],
    }


test_dataset = test_dataset.map(tokenize_function)

# Load all metrics
logger.info("Loading metrics...")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
chrf = evaluate.load("chrf")


# Function to generate predictions
def generate_predictions(model, tokenizer, dataset):
    model.eval()
    predictions = []
    references = []
    references_for_bleu = []  # BLEU expects a list of references per prediction

    logger.info("Generating predictions...")
    for example in tqdm(dataset, desc="Processing", unit="sample"):
        input_text = example["Instructions"] + " " + example["Input"]
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=200,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(generated_text)
        references.append(example["Output"])
        references_for_bleu.append([example["Output"]])  # Wrap in list for BLEU

    return predictions, references, references_for_bleu


# Generate outputs (do this only once for all metrics)
logger.info("Generating model outputs...")
preds, refs, refs_for_bleu = generate_predictions(model, tokenizer, test_dataset)

# Compute all metrics
logger.info("Computing metrics...")
bleu_scores = bleu.compute(predictions=preds, references=refs_for_bleu)
rouge_scores = rouge.compute(predictions=preds, references=refs)
chrf_scores = chrf.compute(predictions=preds, references=refs)

# Combine results
results = {
    "BLEU": bleu_scores["bleu"],
    "ROUGE-1": rouge_scores["rouge1"],
    "ROUGE-2": rouge_scores["rouge2"],
    "ROUGE-L": rouge_scores["rougeL"],
    "ChRF": chrf_scores["score"],
}

# Log and save results
logger.info("\nFinal Test Metrics:")
for key, value in results.items():
    logger.info(f"{key}: {value:.4f}")

with open(OUTPUT_FILE, "w") as f:
    f.write("Final Test Metrics:\n")
    for key, value in results.items():
        f.write(f"{key}: {value:.4f}\n")

logger.info(f"\nResults saved in: {OUTPUT_FILE}")

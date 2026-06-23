import os, torch, numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from datasets import DatasetDict
import evaluate
from sklearn.utils.class_weight import compute_class_weight
from data.load_datasets import build_dataset, ID2LABEL, LABEL2ID, NUM_LABELS
from data.preprocess import balance_classes, split_dataset

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = "./model/checkpoints"
MAX_LEN    = 256   # 256 covers 99% of prompts; 512 if you have GPU RAM


# ── 1. Load and prepare data ───────────────────────────────────
def prepare_data():
    raw_df  = build_dataset()
    bal_df  = balance_classes(raw_df, max_per_class=2000)
    datasets = split_dataset(bal_df)
    return datasets, bal_df["label"].values


# ── 2. Tokenize ───────────────────────────────────────────────
def tokenize_fn(tokenizer, examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )


# ── 3. Custom Trainer with class-weighted loss ─────────────────
class WeightedTrainer(Trainer):
    """Override loss to handle class imbalance."""
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── 4. Metrics ─────────────────────────────────────────────────
def make_compute_metrics():
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1_macro = metric.compute(
            predictions=preds, references=labels, average="macro"
        )["f1"]
        f1_weighted = metric.compute(
            predictions=preds, references=labels, average="weighted"
        )["f1"]
        acc = (preds == labels).mean()
        return {
            "f1_macro":    round(f1_macro,    4),
            "f1_weighted": round(f1_weighted, 4),
            "accuracy":    round(acc,          4),
        }
    return compute_metrics


# ── 5. Main training loop ──────────────────────────────────────
def train():
    print("🚀 Preparing data...")
    datasets, all_labels = prepare_data()

    print("🔡 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    tokenized = datasets.map(
        lambda ex: tokenize_fn(tokenizer, ex),
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format("torch")

    print("🤖 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Compute class weights for imbalanced data
    classes      = np.unique(all_labels)
    weights_np   = compute_class_weight("balanced", classes=classes, y=all_labels)
    class_weights = torch.tensor(weights_np, dtype=torch.float32)
    print(f"⚖️  Class weights: {dict(zip(ID2LABEL.values(), weights_np.round(3)))}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",      # swap to "wandb" if you want W&B logging
        fp16=torch.cuda.is_available(),  # Mixed precision on GPU
        dataloader_num_workers=2,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("🔥 Starting training...")
    trainer.train()

    print("💾 Saving best model...")
    trainer.save_model(OUTPUT_DIR + "/best_model")
    tokenizer.save_pretrained(OUTPUT_DIR + "/best_model")
    print("✅ Training complete!")

    # Final eval on test set
    results = trainer.evaluate(tokenized["test"])
    print(f"\n📊 Test results: {results}")
    return trainer


if __name__ == "__main__":
    train()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_w2v_bert.py
-----------------
Fine-tune facebook/w2v-bert-2.0 on Maltese ASR using CTC.
Integrates old training with specific W2V-BERT feature extraction.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import logging
import multiprocessing
import re
import sys
import time
import traceback
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import librosa
from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
)

logger = logging.getLogger("w2v-bert-train")

# ──────────────────────────────────────────────
# 1. Constants & Maltese Tokenizer Setup
# ──────────────────────────────────────────────
SAMPLING_RATE = 16_000


# ──────────────────────────────────────────────
# 2. Text Normalization 
# ──────────────────────────────────────────────
def normalize_maltese_text(text: str) -> str:
    """
    Normalization logic used in ScriptTest.py.
    Removes specific punctuation, maps rare accents, and uses MASRI tokenizer.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Initial cleaning: remove digits, underscores, backticks, and specific punctuation
    #
    chars_to_remove_regex = r'[\,\?\.\!\;\:\"\“\%\‘\”\\»\«\d\_\`\–]'
    text = re.sub(chars_to_remove_regex, '', text).lower()

    # 2. Character Mapping
    mapping = {
        'á': 'a',
        'é': 'e',
        'ć': 'ċ',
        'í': 'i',
        'ó': 'o', 
        'ʼ': "'", 
        "’": "'"
    }

    for char, replacement in mapping.items():
        text = text.replace(char, replacement)

    # Cleaning white space
    text = " ".join(text.split())
        
    return text

# ──────────────────────────────────────────────
# 3. Vocab & Dataset Builders
# ──────────────────────────────────────────────
def _load_tsv(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    if not {"audio", "text"}.issubset(df.columns):
        raise ValueError(f"Expected 'audio' and 'text' columns in {tsv_path}")
    df = df.rename(columns={"audio": "audio_path", "text": "transcription"})
    return df[["audio_path", "transcription"]].dropna()

def build_vocab(train_tsv: str, dev_tsv: str, output_dir: Path) -> Path:
    vocab_path = output_dir / "vocab.json"
    if vocab_path.exists():
        return vocab_path

    chars = set()
    for tsv in [train_tsv, dev_tsv]:
        df = _load_tsv(tsv)
        for text in df["transcription"]:
            chars.update(normalize_maltese_text(text))

    chars.discard(" ")
    vocab_dict = {c: i for i, c in enumerate(sorted(list(chars)))}
    vocab_dict["|"] = len(vocab_dict) # Word delimiter
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    return vocab_path

def load_and_prepare_dataset(
    tsv_path: str,
    audio_root: Optional[str],
    processor: Wav2Vec2BertProcessor,
    num_proc: int = 16,
) -> Dataset:
    df = _load_tsv(tsv_path)
    if audio_root:
        root = Path(audio_root)
        df["audio_path"] = df["audio_path"].apply(
            lambda p: str(root / p) if not Path(p).is_absolute() else p
        )

    hf_dataset = Dataset.from_pandas(df)

    def prepare_batch(batch):
        try:
            # Using librosa as per ScriptTest.py
            speech, _ = librosa.load(batch["audio_path"], sr=SAMPLING_RATE)
        except Exception as e:
            logger.warning(f"Error loading {batch['audio_path']}: {e}")
            return {"input_features": None, "labels": None}

        # W2V-BERT uses input_features (Log-Mel)
        batch["input_features"] = processor(speech, sampling_rate=SAMPLING_RATE).input_features[0]
        
        # Process text labels
        clean_text = normalize_maltese_text(batch["transcription"])
        batch["labels"] = processor(text=clean_text).input_ids
        return batch

    return hf_dataset.map(
        prepare_batch, 
        remove_columns=hf_dataset.column_names, 
        num_proc=num_proc
    ).filter(lambda x: x["input_features"] is not None)

# ──────────────────────────────────────────────
# 4. Data Collator & Metrics
# ──────────────────────────────────────────────
@dataclass
class DataCollatorCTCWithPadding:
    """Collator adapted for W2V-BERT input_features"""
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # Replace padding with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def build_compute_metrics(processor):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str)
        }
    return compute_metrics

# ──────────────────────────────────────────────
# 5. Main Execution
# ──────────────────────────────────────────────
def main(args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Build Vocab & Processor
    vocab_path = build_vocab(args.train_tsv, args.dev_tsv, output_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print("\n" + "="*40)
    print(f"MODEL VOCABULARY ({len(vocab)} tokens):")
    print(json.dumps(vocab, ensure_ascii=False, indent=2))
    print("="*40 + "\n")

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    # W2V-BERT specific feature extractor
    feature_extractor = SeamlessM4TFeatureExtractor(
        feature_size=80, num_mel_bins=80, sampling_rate=SAMPLING_RATE, padding_value=0.0
    )
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 2. Load Model
    model = Wav2Vec2BertForCTC.from_pretrained(
        "facebook/w2v-bert-2.0",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    # 3. Load Data
    train_dataset = load_and_prepare_dataset(args.train_tsv, args.audio_root, processor)
    dev_dataset = load_and_prepare_dataset(args.dev_tsv, args.audio_root, processor)


    # DATA SANITY CHECK 
    print("\n" + "="*40)
    print("DATA SANITY CHECK")
    sample = train_dataset[0]
    decoded_text = processor.decode(sample["labels"])
    print(f"Sample 1 - Processed Transcription: {decoded_text}")
    print(f"Sample 1 - Input Features Shape: {torch.tensor(sample['input_features']).shape}")
    print("="*40 + "\n")

    # 4. Training Arguments 
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=500,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        fp16=True,
        max_grad_norm=1.0,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv", type=str, required=True)
    parser.add_argument("--dev_tsv", type=str, required=True)
    parser.add_argument("--audio_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_steps", type=int, default=300)
    args = parser.parse_args()
    main(args)
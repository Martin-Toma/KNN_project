# -*- coding: utf-8 -*-
"""
finetune_lora_heads.py
LoRA finetuning of google/gemma-3-4b-pt with LM + rating & genre heads

Author: Martin Tomasovic, Attila Kovacs
"""

import json
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA finetuning of Gemma-3-4B with multi-task heads"
    )
    p.add_argument("modelSavePth", help="Directory to save the fine-tuned model")
    p.add_argument("--r",      type=int,   default=32,     help="LoRA rank")
    p.add_argument("--alpha",  type=int,   default=64,    help="LoRA alpha")
    p.add_argument("--dropout",type=float, default=0.1,  help="LoRA dropout")
    p.add_argument("--lr",     type=float, default=5e-4,  help="Learning rate")
    p.add_argument("--alpha_loss", type=float, default=0.1, help="Weight for LM loss")
    p.add_argument("--beta_loss",  type=float, default=10.0, help="Weight for rating loss")
    p.add_argument("--gamma_loss", type=float, default=10.0, help="Weight for genre loss")
    return p.parse_args()


class MultiTaskModel(torch.nn.Module):
    def __init__(self, base_model, num_genres, args):
        super().__init__()
        self.base_model = base_model
        # expose config so PEFT can see it:
        self.config = base_model.config  

        cfg = base_model.config
        inner_cfg = getattr(cfg, "text_config", cfg)

        hidden_size = (
            getattr(inner_cfg, "hidden_size", None)
            or getattr(inner_cfg, "d_model", None)
            or getattr(inner_cfg, "n_embd",  None)
        )
        if hidden_size is None:
            emb = base_model.get_input_embeddings()
            if hasattr(emb, "embedding_dim"):
                hidden_size = emb.embedding_dim

        if hidden_size is None:
            raise ValueError("Could not infer hidden size from Gemma-3 config")

        self.rating_head = torch.nn.Linear(hidden_size, 1)
        self.genre_head  = torch.nn.Linear(hidden_size, num_genres)
        self.args = args

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        rating_labels=None,
        genre_labels=None,
        **kwargs
    ):
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        last_hidden = out.hidden_states[-1][
            torch.arange(last_idx.size(0)), last_idx
        ]

        rating_logits = self.rating_head(last_hidden).squeeze(-1)
        genre_logits  = self.genre_head(last_hidden)

        lm_loss  = out.loss
        reg_loss = torch.nn.functional.huber_loss(
            rating_logits, rating_labels, delta=1.0
        )
        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            genre_logits, genre_labels.float()
        )

        total_loss = (
            self.args.alpha_loss * lm_loss
            + self.args.beta_loss  * reg_loss
            + self.args.gamma_loss * cls_loss
        )

        return {
            "loss": total_loss,
            "lm_loss": lm_loss.detach(),
            "regression_loss": reg_loss.detach(),
            "classification_loss": cls_loss.detach(),
        }

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return self.base_model.prepare_inputs_for_generation(
            input_ids, attention_mask=attention_mask, **kwargs
        )


class MultiTaskCollator:
    def __call__(self, features):
        return {
            "input_ids":      torch.stack([f["input_ids"]      for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels":         torch.stack([f["labels"]         for f in features]),
            "rating_labels":  torch.stack([f["rating_label"]   for f in features]),
            "genre_labels":   torch.stack([f["genre_labels"]   for f in features]),
        }


def main():
    args = parse_args()

    with open("genre2id.json", "r") as f:
        genre2id = json.load(f)
    num_genres = len(genre2id)

    model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    model = MultiTaskModel(base_model, num_genres, args)

    peft_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    train_ds = load_from_disk("g32_head_train_tokenized_8k")
    val_ds   = load_from_disk("g32_head_validation_tokenized_8k")
    for ds in (train_ds, val_ds):
        ds.set_format(
            type="torch",
            columns=["input_ids","attention_mask","labels","rating_label","genre_labels"]
        )

    training_args = TrainingArguments(
        output_dir=args.modelSavePth,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        num_train_epochs=1,
        do_eval=False,
        max_steps=1250, # Due to time constraints, and high VRAM use(>40GB) during training we had to limit the training to the first 10000 samples.
        fp16=True,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=MultiTaskCollator(),
    )

    trainer.train()
    trainer.save_model(args.modelSavePth)


if __name__ == "__main__":
    main()

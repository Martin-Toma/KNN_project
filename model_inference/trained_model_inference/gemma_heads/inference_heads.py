#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import re
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import argparse
from jinja2 import Environment, BaseLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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


def clean_subtitle(text: str) -> str:
    text = re.sub(
        r'(?m)(?:^[0-9]+\r?\n|^\d{2}:\d{2}:\d{2},\d{3} -->.*\r?\n|<[^>]+>)',
        "",
        text,
    )
    text = re.sub(r'(?m)^\s*\r?\n', "", text)
    return text.strip()[:8000]


def load_genre2id(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Fixed inference with LoRA-finetuned MultiTaskModel"
    )
    parser.add_argument("input_file", help="Path to subtitle .srt file")
    parser.add_argument("--base_model", default="google/gemma-3-4b-pt")
    parser.add_argument("--lora_dir",   default="./lora_heads")
    parser.add_argument("--genre2id",   default="genre2id.json")
    parser.add_argument("--template",   default="chat_template.json")
    args = parser.parse_args()

    # 1) Tokenizer & genre mappings
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    genre2id = load_genre2id(args.genre2id)
    id2genre = {v: k for k, v in genre2id.items()}

    # 2) Load the base LM onto GPU/CPU in bfloat16
    base_lm = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3) Wrap in your MultiTaskModel
    dummy_args = argparse.Namespace(alpha_loss=1.0, beta_loss=1.0, gamma_loss=1.0)
    wrapper = MultiTaskModel(base_lm, len(genre2id), dummy_args)

    # 4) Inject LoRA adapters onto that wrapper
    model = PeftModel.from_pretrained(
        wrapper,
        args.lora_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 5) Unwrap the HF + LoRA LM so we can call generate()
    hf_lm = model.model.base_model

    # 6) Ensure generation_config is set
    hf_lm.generation_config = hf_lm.generation_config

    model.eval()

    # ─────────────── DEVICE & DTYPE SYNC ───────────────
    # Move + cast your custom heads to the same device & dtype as hf_lm
    device = next(hf_lm.parameters()).device
    dtype  = next(hf_lm.parameters()).dtype

    model.model.rating_head.to(device).to(dtype)
    model.model.genre_head.to(device).to(dtype)
    # ────────────────────────────────────────────────────

    # 7) Read & clean subtitles
    with open(args.input_file, "r", encoding="utf-8") as f:
        subtitle = clean_subtitle(f.read())

    # 8) Build JSON-only prompt
    with open(args.template, "r", encoding="utf-8") as f:
        tmpl_json = json.load(f)["chat_template"]
    env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    tmpl = env.from_string(tmpl_json)

    system_msg = {
        "role": "system",
        "content": (
            "You only respond with JSON and nothing else. "
            "When given a subtitle, write a short review and guess the genre(s). "
            "Output exactly one JSON object with keys: review (string), "
            "genres (array of strings), rating (0–10)."
        )
    }
    prompt = tmpl.render(
        messages=[system_msg, {"role": "user", "content": subtitle}],
        add_generation_prompt=True,
    )

    # 9) Tokenize & move to model device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 10) Generate the LM’s JSON blob via hf_lm.generate(...)
    with torch.no_grad():
        gen_ids = hf_lm.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    full_out = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    j0, j1 = full_out.find("{"), full_out.rfind("}") + 1
    try:
        parsed = json.loads(full_out[j0:j1])
    except json.JSONDecodeError:
        print("Error: invalid JSON\n", full_out, file=sys.stderr)
        sys.exit(1)

    # 11) Run your custom heads on the same last hidden state
    with torch.no_grad():
        lm_out = hf_lm(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        last_idx = (inputs["attention_mask"].sum(dim=1) - 1).clamp(min=0)
        last_h = lm_out.hidden_states[-1][
            torch.arange(last_idx.size(0)), last_idx
        ]

        # heads are now on the right device & dtype
        r_logit = model.model.rating_head(last_h).squeeze(-1)
        g_logit = model.model.genre_head(last_h)

        rating_pred = r_logit.item()
        probs = torch.sigmoid(g_logit)[0].tolist()
        positives = [id2genre[i] for i, p in enumerate(probs) if p >= 0.8]

    # 12) Merge & print
    parsed["head_rating"] = round(rating_pred, 2)
    parsed["head_genres_predicted"] = positives
    parsed["head_genres_probs"] = {
        id2genre[i]: round(p, 3) for i, p in enumerate(probs)
    }

    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()


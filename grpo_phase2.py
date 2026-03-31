# """
# SPTAR Phase 2: GRPO Soft-Prompt Improvement Loop
# =================================================
# Fixes applied:
#   1. DPR SentenceTransformer forced to CPU (avoids CUDA driver error)
#   2. get_ref_log_prob uses CPU tensors to avoid CUDACachingAllocator crash
#   4. torch_dtype -> dtype deprecation fixed
#   5. ref_encoder_state swap done safely without touching GPU allocator
# """

# import os, sys, json, csv, argparse, random, logging, math, warnings
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# import torch
# import torch.nn.functional as F
# import numpy as np
# from torch.optim import AdamW
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     get_linear_schedule_with_warmup,
# )
# from peft import PeftConfig

# REPO_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(REPO_ROOT / "xuyang"))
# sys.path.insert(0, str(REPO_ROOT / "zhiyuan"))

# from default_prompt import DefaultPrompt

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )
# logger = logging.getLogger(__name__)


# # ─────────────────────────────────────────────────────────────────────────────
# # 1. ARGS
# # ─────────────────────────────────────────────────────────────────────────────

# def parse_args():
#     p = argparse.ArgumentParser("SPTAR Phase 2 - GRPO soft-prompt improvement")

#     p.add_argument("--peft_model_id", required=True)
#     p.add_argument("--dpr_ckpt", required=True)
#     p.add_argument("--weak_queries", required=True)
#     p.add_argument("--weak_qrels", required=True)
#     p.add_argument("--corpus", required=True)
#     p.add_argument("--eval_corpus", required=True)
#     p.add_argument("--dev_queries", required=True)
#     p.add_argument("--dev_qrels", required=True)
#     p.add_argument("--output_dir", default="output/grpo_phase2")

#     p.add_argument("--dataset_name", default="fiqa_50",
#                    choices=["fiqa_50", "ms_50", "hotpotqa_50", "fever_50"])
#     p.add_argument("--prompt_num", type=int, default=2)
#     p.add_argument("--text_len", type=int, default=350)
#     p.add_argument("--train_num", type=int, default=50)

#     p.add_argument("--load_in_4bit", action="store_true", default=True)
#     p.add_argument("--base_model", type=str, default=None)

#     p.add_argument("--grpo_group_size", type=int, default=4)
#     p.add_argument("--lr", type=float, default=3e-2)
#     p.add_argument("--grpo_epochs", type=int, default=3)
#     p.add_argument("--batch_docs", type=int, default=1)
#     p.add_argument("--kl_coeff", type=float, default=0.04)
#     p.add_argument("--clip_ratio", type=float, default=0.2)

#     p.add_argument("--w_relevance",      type=float, default=0.4)
#     p.add_argument("--w_retrievability", type=float, default=0.4)
#     p.add_argument("--w_specificity",    type=float, default=0.2)

#     p.add_argument("--dpr_retrain_every",  type=int, default=200)
#     p.add_argument("--dpr_retrain_epochs", type=int, default=2)
#     p.add_argument("--dpr_batch_size",     type=int, default=16)

#     p.add_argument("--patience",       type=int,   default=4)
#     p.add_argument("--ndcg_min_delta", type=float, default=0.001)

#     p.add_argument("--device",         type=str,   default="cuda:0")
#     p.add_argument("--max_new_tokens", type=int,   default=64)
#     p.add_argument("--temperature",    type=float, default=0.7)
#     p.add_argument("--seed",           type=int,   default=42)

#     return p.parse_args()


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. DATA UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────

# def load_jsonl(path: str) -> List[Dict]:
#     out = []
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 out.append(json.loads(line))
#     return out


# def save_weak_queries_jsonl(queries: List[Dict], path: str):
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     with open(path, "w") as f:
#         for q in queries:
#             json.dump(q, f)
#             f.write("\n")


# def save_weak_qrels_tsv(qrels: List[Tuple], path: str):
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     with open(path, "w", newline="") as f:
#         w = csv.writer(f, delimiter="\t")
#         w.writerow(["query-id", "corpus-id", "score"])
#         for row in qrels:
#             w.writerow(row)


# def load_corpus(path: str) -> List[Dict]:
#     ext = Path(path).suffix.lower()
#     if ext == ".csv":
#         import pandas as pd
#         df = pd.read_csv(path)
#         corpus = []
#         for _, row in df.iterrows():
#             corpus.append({
#                 "_id":   str(row["_id"]),
#                 "title": str(row.get("title", "")),
#                 "text":  str(row["text"]),
#             })
#     else:
#         corpus = load_jsonl(path)
#     logger.info(f"Corpus: {len(corpus)} documents from {path}")
#     return corpus


# def load_phase1_weak_data(queries_path: str, qrels_path: str
#                           ) -> Tuple[List[Dict], List[Tuple]]:
#     queries = load_jsonl(queries_path)
#     qrels: List[Tuple] = []
#     with open(qrels_path) as f:
#         reader = csv.DictReader(f, delimiter="\t")
#         for row in reader:
#             qrels.append((row["query-id"], row["corpus-id"], row["score"]))
#     logger.info(f"Phase 1 data: {len(queries)} queries, {len(qrels)} qrels")
#     return queries, qrels


# def cut_text(text: str, max_len: int = 350) -> str:
#     words = text.split()
#     return " ".join(words[:max_len]) if len(words) > max_len else text


# # ─────────────────────────────────────────────────────────────────────────────
# # 3. PROMPT BUILDER
# # ─────────────────────────────────────────────────────────────────────────────

# def get_fixed_prompts(dataset_name: str) -> Tuple[str, str]:
#     if "ms_" in dataset_name:
#         return (DefaultPrompt.ms_50_fixed_one_shot_prompt,
#                 DefaultPrompt.ms_50_fixed_two_shot_prompt)
#     if dataset_name == "fiqa_50":
#         return (DefaultPrompt.fiqa_50_fixed_one_shot_prompt,
#                 DefaultPrompt.fiqa_50_fixed_two_shot_prompt)
#     if dataset_name == "hotpotqa_50":
#         return (DefaultPrompt.hotpotqa_50_fixed_one_shot_prompt,
#                 DefaultPrompt.hotpotqa_50_fixed_two_shot_prompt)
#     if dataset_name == "fever_50":
#         return (DefaultPrompt.fever_50_fixed_one_shot_prompt,
#                 DefaultPrompt.fever_50_fixed_two_shot_prompt)
#     raise ValueError(f"Unknown dataset_name: {dataset_name}")


# def build_prompt(corpus_text: str, prompt_num: int,
#                  one_shot: str, two_shot: str,
#                  text_len: int = 350) -> str:
#     corpus_text = cut_text(corpus_text, text_len)
#     if prompt_num == 2:
#         return "{} \n Document: {} \n Relevant Query: ".format(one_shot, corpus_text)
#     elif prompt_num == 3:
#         return "{} \n Document: {} \n Relevant Query: ".format(two_shot, corpus_text)
#     else:
#         return "Document: {} \n Relevant Query: ".format(corpus_text)


# def simple_filter(text: str) -> str:
#     text = text.split("\n")[0]
#     for punt in [".", ",", "?"]:
#         pre_i, new_text = "", ""
#         for i in text.split(punt):
#             if pre_i != i:
#                 new_text += i
#                 pre_i = i
#             else:
#                 break
#         text = new_text
#     return text.strip()


# # ─────────────────────────────────────────────────────────────────────────────
# # 4. PEFT MODEL WRAPPER
# # ─────────────────────────────────────────────────────────────────────────────

# class PeftSoftPromptModel:

#     def __init__(self, peft_model_id: str, device: torch.device,
#                  load_in_4bit: bool = True, base_model_override: str = None):
#         self.device = device

#         logger.info(f"Loading PEFT config from {peft_model_id}")
#         peft_model_id = str(Path(peft_model_id).resolve())

#         config    = PeftConfig.from_pretrained(peft_model_id)
#         base_path = config.base_model_name_or_path

#         if base_model_override:
#             base_path = base_model_override
#             logger.info(f"Base model overridden to: {base_path}")
#         elif not os.path.exists(base_path):
#             logger.warning(f"base_model_name_or_path {base_path!r} not found locally.")
#             logger.warning("Falling back to meta-llama/Llama-2-7b-hf from HF Hub.")
#             logger.warning("Set --base_model to override if you have a local copy.")
#             base_path = "meta-llama/Llama-2-7b-hf"
#         logger.info(f"Base LLM: {base_path}")

#         # FIX: use dtype= instead of deprecated torch_dtype=
#         if load_in_4bit:
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_compute_dtype=torch.float16,
#             )
#             base = AutoModelForCausalLM.from_pretrained(
#                 base_path,
#                 quantization_config=bnb_config,
#                 device_map={"":0},
#                 dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#             )
#             logger.info("Base LLM loaded in 4-bit NF4")
#         else:
#             base = AutoModelForCausalLM.from_pretrained(
#                 base_path,
#                 dtype=torch.float16,
#                 device_map={"": device},
#             )

#         from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

#         num_virtual_tokens = getattr(config, "num_virtual_tokens", 50)
#         init_text = getattr(config, "prompt_tuning_init_text",
#                             "please generate query for this document")

#         logger.info(f"Creating fresh PEFT adapter: "
#                     f"num_virtual_tokens={num_virtual_tokens}, "
#                     f"init_text='{init_text}'")

#         peft_config = PromptTuningConfig(
#             task_type="CAUSAL_LM",
#             prompt_tuning_init=PromptTuningInit.TEXT,
#             prompt_tuning_init_text=init_text,
#             num_virtual_tokens=num_virtual_tokens,
#             tokenizer_name_or_path=base_path,
#         )
#         self.model = get_peft_model(base, peft_config)
#         logger.info("Fresh PEFT prompt tuning adapter created — "
#                     "will train from scratch via GRPO")

#         for name, param in self.model.named_parameters():
#             if "prompt_encoder" in name:
#                 param.requires_grad = True
#                 param.data = param.data.float()
#             else:
#                 param.requires_grad = False

#         n_train = sum(p.numel() for p in self.model.parameters()
#                       if p.requires_grad)
#         logger.info(f"Trainable params: {n_train:,} (prompt encoder only)")

#         self.tokenizer = AutoTokenizer.from_pretrained(base_path)
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

#     def save(self, path: str):
#         self.model.save_pretrained(path)
#         logger.info(f"Saved PEFT adapter -> {path}")

#     @torch.no_grad()
#     def generate_query(self, prompt: str,
#                        max_new_tokens: int = 64,
#                        temperature: float = 0.7) -> str:
#         inputs = self.tokenizer(
#             prompt, return_tensors="pt",
#             truncation=True, max_length=512
#         )
#         inputs = {k: v.to(next(self.model.parameters()).device)
#                   for k, v in inputs.items()}
#         with torch.amp.autocast("cuda", dtype=torch.float16):
#             outputs = self.model.generate(
#                 input_ids=inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 max_new_tokens=max_new_tokens,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 temperature=temperature,
#                 repetition_penalty=1.3,
#                 no_repeat_ngram_size=3,
#                 do_sample=True,
#             )
#         full_text = self.tokenizer.batch_decode(
#             outputs.detach().cpu().numpy(), skip_special_tokens=True
#         )[0]
#         query = full_text[len(prompt):]
#         del outputs
#         torch.cuda.empty_cache()
#         return simple_filter(query)

#     @torch.no_grad()
#     def generate_group(self, prompt: str, G: int,
#                        max_new_tokens: int, temperature: float) -> List[str]:
#         results = []
#         for _ in range(G):
#             q = self.generate_query(prompt, max_new_tokens, temperature)
#             results.append(q)
#         return results

#     def compute_log_prob(self, prompt: str, query: str) -> torch.Tensor:
#         torch.cuda.empty_cache()

#         full_text  = prompt + query
#         prompt_ids = self.tokenizer(
#             prompt, return_tensors="pt",
#             truncation=True, max_length=512
#         ).input_ids.to(self.device)
#         full_ids = self.tokenizer(
#             full_text, return_tensors="pt",
#             truncation=True, max_length=576
#         ).input_ids.to(self.device)

#         answer_len = full_ids.shape[1] - prompt_ids.shape[1]
#         if answer_len <= 0:
#             return sum(p.sum() * 0.0
#                        for p in self.model.parameters()
#                        if p.requires_grad)

#         with torch.amp.autocast("cuda", dtype=torch.float16):
#             out = self.model(
#                 input_ids=full_ids,
#                 attention_mask=torch.ones_like(full_ids),
#             )

#         prompt_len = prompt_ids.shape[1]
#         logits     = out.logits[0].float()
#         pred       = logits[prompt_len - 1: prompt_len - 1 + answer_len]
#         targets    = full_ids[0, prompt_len: prompt_len + answer_len]

#         n         = min(pred.shape[0], targets.shape[0])
#         log_probs = F.log_softmax(pred[:n], dim=-1)
#         result    = log_probs[torch.arange(n, device=self.device), targets[:n]].sum()

#         del out, logits, pred, log_probs
#         torch.cuda.empty_cache()

#         return result


# # ─────────────────────────────────────────────────────────────────────────────
# # 5. DPR MANAGER
# #    FIX: always load SentenceTransformer on CPU to avoid CUDA driver error
# # ─────────────────────────────────────────────────────────────────────────────

# class DPRManager:

#     def __init__(self, dpr_ckpt: str, corpus: List[Dict], device: torch.device):
#         self.device = device
#         self.corpus = corpus
#         self._load_model(dpr_ckpt)
#         self._build_index()

#     def _load_model(self, path: str):
#         try:
#             from sentence_transformers import SentenceTransformer
#             # FIX: force CPU for DPR — avoids CUDA driver error from WSL2
#             # device nodes issue. CPU is fast enough for 500-doc corpus.
#             self.model = SentenceTransformer(path, device="cpu")
#             logger.info(f"DPR loaded from {path} (on CPU)")
#         except Exception as e:
#             logger.warning(f"DPR load failed ({e}). "
#                            "Retrievability reward will return 0.5.")
#             self.model = None

#     def _build_index(self):
#         if self.model is None:
#             self.corpus_embeddings = None
#             self.corpus_ids        = []
#             return
#         logger.info("Building DPR dense index ...")
#         self.corpus_ids   = [d["_id"]  for d in self.corpus]
#         corpus_texts      = [d["text"] for d in self.corpus]
#         self.corpus_embeddings = self.model.encode(
#             corpus_texts,
#             batch_size=32,
#             show_progress_bar=True,
#             convert_to_tensor=True,
#             device="cpu",
#             normalize_embeddings=True,
#         )
#         logger.info(f"DPR index shape: {self.corpus_embeddings.shape}")

#     def retrieve(self, query: str, top_k: int = 10) -> List[str]:
#         if self.model is None or self.corpus_embeddings is None:
#             return []
#         q_emb = self.model.encode(
#             query, convert_to_tensor=True,
#             device="cpu", normalize_embeddings=True
#         )
#         # both on CPU — safe matmul
#         scores  = torch.matmul(self.corpus_embeddings, q_emb)
#         top_idx = scores.topk(min(top_k, len(self.corpus_ids))).indices.tolist()
#         return [self.corpus_ids[i] for i in top_idx]

#     def retrain(self, all_queries: List[Dict], all_qrels: List[Tuple],
#                 args) -> None:
#         import subprocess, shutil

#         dataset_core  = args.dataset_name.split("_")[0]
#         weak_num_str  = str(len(all_queries))
#         exp_tag       = "grpo_phase2"
#         xuyang_dir    = str(REPO_ROOT / "xuyang" / "data")
#         target_dir    = os.path.join(xuyang_dir, args.dataset_name, weak_num_str)
#         os.makedirs(target_dir, exist_ok=True)

#         wq_path = os.path.join(target_dir,
#                                f"weak_queries_{args.train_num}_{exp_tag}.jsonl")
#         wt_path = os.path.join(target_dir,
#                                f"weak_train_{args.train_num}_{exp_tag}.tsv")
#         save_weak_queries_jsonl(all_queries, wq_path)
#         save_weak_qrels_tsv(all_qrels, wt_path)
#         logger.info(f"Wrote {len(all_queries)} queries -> {wq_path}")

#         beir_dir = os.path.join(str(REPO_ROOT / "zhiyuan"),
#                                 "datasets", "raw", "beir", dataset_core)
#         eval_corpus_dest = os.path.join(
#             beir_dir, f"corpus_{weak_num_str}_reduced_ratio_20.jsonl")
#         if not os.path.exists(eval_corpus_dest):
#             shutil.copy(args.eval_corpus, eval_corpus_dest)
#             logger.info(f"Copied eval corpus -> {eval_corpus_dest}")

#         train_script = str(REPO_ROOT / "zhiyuan" / "retriever" / "dpr"
#                            / "train" / "train_sbert.py")
#         cmd = [
#             sys.executable, train_script,
#             "--dataset_name", dataset_core,
#             "--num_epochs",   str(args.dpr_retrain_epochs),
#             "--train_num",    str(args.train_num),
#             "--weak_num",     weak_num_str,
#             "--exp_name",     exp_tag,
#         ]
#         logger.info(f"DPR retrain command:\n  {' '.join(cmd)}")
#         ret = subprocess.run(cmd)
#         if ret.returncode != 0:
#             logger.error("DPR retrain subprocess failed - keeping old model.")
#             return

#         model_name   = "bert-large-uncased"
#         new_dpr_path = os.path.join(
#             str(REPO_ROOT / "zhiyuan"), "retriever", "dpr", "train", "output",
#             exp_tag, str(args.train_num),
#             f"{model_name}-v1-{dataset_core}"
#         )
#         if os.path.exists(new_dpr_path):
#             self._load_model(new_dpr_path)
#             self._build_index()
#             logger.info(f"DPR reloaded from {new_dpr_path}")
#         else:
#             logger.warning(f"DPR output path not found: {new_dpr_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # 6. REWARD FUNCTIONS
# # ─────────────────────────────────────────────────────────────────────────────

# def reward_relevance(query: str, doc_text: str) -> float:
#     if not query.strip():
#         return 0.0
#     q_toks = set(query.lower().split())
#     d_toks = set(doc_text.lower().split())
#     if not q_toks:
#         return 0.0
#     overlap      = len(q_toks & d_toks) / len(q_toks)
#     length_bonus = min(1.0, len(query.split()) / 8.0)
#     return 0.6 * overlap + 0.4 * length_bonus


# def reward_specificity(query: str) -> float:
#     q = query.lower().strip()
#     if not q:
#         return 0.0
#     score = 1.0
#     if any(q.startswith(g) for g in
#            ["is ", "are ", "was ", "were ", "do ", "does ",
#             "did ", "can ", "could ", "would ", "should "]):
#         score *= 0.6
#     if any(q.startswith(w) for w in
#            ["what", "who", "when", "where", "why", "how", "which"]):
#         score = min(1.0, score * 1.3)
#     n = len(q.split())
#     if n < 4:
#         score *= 0.3
#     elif n > 25:
#         score *= 0.7
#     return min(1.0, score)


# def reward_retrievability(query: str, corpus_id: str,
#                           dpr: DPRManager, top_k: int = 10) -> float:
#     if dpr.model is None:
#         return 0.5
#     results = dpr.retrieve(query, top_k=top_k)
#     if corpus_id in results:
#         rank = results.index(corpus_id) + 1
#         return 1.0 / math.log2(rank + 1)
#     return 0.0


# def composite_reward(query: str, doc_text: str, corpus_id: str,
#                      dpr: DPRManager, args) -> float:
#     return (args.w_relevance      * reward_relevance(query, doc_text)
#           + args.w_retrievability * reward_retrievability(query, corpus_id, dpr)
#           + args.w_specificity    * reward_specificity(query))


# # ─────────────────────────────────────────────────────────────────────────────
# # 7. NDCG@10
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_ndcg_at_10(dpr: DPRManager,
#                        eval_pairs: List[Tuple[str, str]],
#                        top_k: int = 10) -> float:
#     if dpr.model is None or not eval_pairs:
#         return 0.0
#     scores = []
#     for query_text, corpus_id in eval_pairs:
#         results = dpr.retrieve(query_text, top_k=top_k)
#         if corpus_id in results:
#             rank = results.index(corpus_id) + 1
#             scores.append(1.0 / math.log2(rank + 1))
#         else:
#             scores.append(0.0)
#     return float(np.mean(scores))


# # ─────────────────────────────────────────────────────────────────────────────
# # 8. GRPO LOSS
# # ─────────────────────────────────────────────────────────────────────────────

# def grpo_loss(curr_log_probs: List[torch.Tensor],
#               ref_log_probs:  List[float],
#               rewards:        List[float],
#               kl_coeff:       float,
#               clip_ratio:     float) -> torch.Tensor:
#     if not rewards:
#         return torch.tensor(0.0, requires_grad=True)

#     r_t   = torch.tensor(rewards, dtype=torch.float32)
#     adv_t = (r_t - r_t.mean()) / (r_t.std() + 1e-8)

#     losses = []
#     for lp_curr, lp_ref, adv in zip(curr_log_probs, ref_log_probs, adv_t):
#         adv      = adv.to(lp_curr.device)
#         ratio    = torch.exp(lp_curr - float(lp_ref))
#         clipped  = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
#         pol_loss = -torch.min(ratio * adv, clipped * adv)
#         kl_pen   = kl_coeff * (lp_curr - float(lp_ref))
#         losses.append(pol_loss + kl_pen)

#     return torch.stack(losses).mean()


# # ─────────────────────────────────────────────────────────────────────────────
# # 9. MAIN TRAINING LOOP
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     args = parse_args()

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)


#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     logger.info(f"Device: {device}")
#     os.makedirs(args.output_dir, exist_ok=True)

#     corpus                       = load_corpus(args.corpus)
#     phase1_queries, phase1_qrels = load_phase1_weak_data(
#         args.weak_queries, args.weak_qrels
#     )

#     random.shuffle(phase1_queries)
#     split        = max(1, int(0.8 * len(phase1_queries)))
#     seed_queries = phase1_queries[:split]
#     eval_queries = phase1_queries[split:]
#     seed_ids     = {q["_id"] for q in seed_queries}
#     seed_qrels   = [qr for qr in phase1_qrels if qr[0] in seed_ids]

#     qid_to_cid  = {qr[0]: qr[1] for qr in phase1_qrels}
#     eval_pairs: List[Tuple[str, str]] = [
#         (q["text"], qid_to_cid[q["_id"]])
#         for q in eval_queries
#         if q["_id"] in qid_to_cid
#     ]
#     logger.info(f"Seed: {len(seed_queries)} queries | "
#                 f"Eval: {len(eval_queries)} queries | "
#                 f"Eval pairs: {len(eval_pairs)}")

#     one_shot, two_shot = get_fixed_prompts(args.dataset_name)

#     peft_model = PeftSoftPromptModel(
#         peft_model_id=args.peft_model_id,
#         device=device,
#         load_in_4bit=args.load_in_4bit,
#         base_model_override=args.base_model,
#     )

#     # FIX 2: save ref encoder state fully on CPU to avoid CUDACachingAllocator crash
#     ref_encoder_state = {
#         k: v.detach().cpu().clone().float()
#         for k, v in peft_model.model.prompt_encoder.state_dict().items()
#     }

#     def get_ref_log_prob(prompt: str, query: str) -> float:
#         # FIX 3: swap encoder weights safely — save live state to CPU first,
#         # load ref state to GPU, compute, restore — all with explicit device moves
#         live_state = {
#             k: v.detach().cpu().clone()
#             for k, v in peft_model.model.prompt_encoder.state_dict().items()
#         }
#         # move ref weights to whichever device the encoder is on
#         enc_device = next(peft_model.model.prompt_encoder.parameters()).device
#         peft_model.model.prompt_encoder.load_state_dict(
#             {k: v.to(enc_device) for k, v in ref_encoder_state.items()}
#         )
#         with torch.no_grad():
#             lp = peft_model.compute_log_prob(prompt, query).item()
#         # restore live weights
#         peft_model.model.prompt_encoder.load_state_dict(
#             {k: v.to(enc_device) for k, v in live_state.items()}
#         )
#         return lp

#     # DPR loads on CPU (fix for CUDA driver error)
#     dpr = DPRManager(args.dpr_ckpt, corpus, device)

#     baseline_ndcg = compute_ndcg_at_10(dpr, eval_pairs)
#     logger.info(f"Baseline NDCG@10 = {baseline_ndcg:.4f}")
#     ndcg_history  = [baseline_ndcg]
#     best_ndcg     = baseline_ndcg
#     no_improve_ct = 0

#     trainable   = [p for p in peft_model.model.parameters() if p.requires_grad]
#     optimizer   = AdamW(trainable, lr=args.lr, weight_decay=0.01)
#     total_steps = (len(corpus) // max(1, args.batch_docs)) * args.grpo_epochs
#     scheduler   = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=max(10, total_steps // 20),
#         num_training_steps=total_steps,
#     )
#     scaler = torch.amp.GradScaler("cuda")

#     all_queries: List[Dict]  = list(seed_queries)
#     all_qrels:   List[Tuple] = list(seed_qrels)
#     next_qid = 3_000_000

#     docs_processed = 0
#     global_step    = 0

#     logger.info("=" * 60)
#     logger.info("Phase 2 GRPO loop starting")
#     logger.info(f"  Corpus:              {len(corpus)} docs")
#     logger.info(f"  Group size G:        {args.grpo_group_size}")
#     logger.info(f"  Docs per step:       {args.batch_docs}")
#     logger.info(f"  Max new tokens:      {args.max_new_tokens}")
#     logger.info(f"  DPR retrain every:   {args.dpr_retrain_every} docs")
#     logger.info(f"  Prompt format:       prompt_num={args.prompt_num}")
#     logger.info("=" * 60)

#     for epoch in range(args.grpo_epochs):
#         random.shuffle(corpus)
#         batches    = [corpus[i: i + args.batch_docs]
#                       for i in range(0, len(corpus), args.batch_docs)]
#         epoch_loss = []

#         for batch in batches:
#             optimizer.zero_grad()
#             batch_loss = None

#             for doc in batch:
#                 corpus_id = doc["_id"]
#                 doc_text  = doc["text"]
#                 prompt    = build_prompt(doc_text, args.prompt_num,
#                                          one_shot, two_shot, args.text_len)

#                 import time
#                 logger.info(f"  [DOC {docs_processed+1}] Starting generation...")
#                 t0 = time.time()
#                 queries_g = peft_model.generate_group(
#                     prompt, args.grpo_group_size,
#                     args.max_new_tokens, args.temperature
#                 )
#                 torch.cuda.empty_cache()
#                 logger.info(f"  [DOC {docs_processed+1}] Generation done in {time.time()-t0:.1f}s | queries: {queries_g}")

#                 queries_g = [q for q in queries_g if q.strip()]
#                 if not queries_g:
#                     logger.info(f"  [DOC {docs_processed+1}] All queries empty, skipping")
#                     continue

#                 t1 = time.time()
#                 rewards_g = [
#                     composite_reward(q, doc_text, corpus_id, dpr, args)
#                     for q in queries_g
#                 ]
#                 logger.info(f"  [DOC {docs_processed+1}] Rewards done in {time.time()-t1:.1f}s | rewards: {rewards_g}")

#                 t2 = time.time()
#                 ref_lps = [get_ref_log_prob(prompt, q) for q in queries_g]
#                 logger.info(f"  [DOC {docs_processed+1}] Ref log probs done in {time.time()-t2:.1f}s")

#                 t3 = time.time()
#                 curr_lps = [peft_model.compute_log_prob(prompt, q)
#                             for q in queries_g]
#                 logger.info(f"  [DOC {docs_processed+1}] Curr log probs done in {time.time()-t3:.1f}s")

#                 doc_loss   = grpo_loss(curr_lps, ref_lps, rewards_g,
#                                        args.kl_coeff, args.clip_ratio)
#                 batch_loss = doc_loss if batch_loss is None \
#                              else batch_loss + doc_loss

#                 threshold = float(np.mean(rewards_g))
#                 for q, r in zip(queries_g, rewards_g):
#                     if r >= threshold:
#                         next_qid += 1
#                         all_queries.append(
#                             {"_id": str(next_qid), "text": q, "metadata": {}}
#                         )
#                         all_qrels.append((str(next_qid), corpus_id, "1"))

#             if batch_loss is None:
#                 continue

#             mean_loss = batch_loss / len(batch)
#             scaler.scale(mean_loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()

#             torch.cuda.empty_cache()

#             epoch_loss.append(mean_loss.item())
#             docs_processed += len(batch)
#             global_step    += 1

#             if global_step % 20 == 0:
#                 logger.info(
#                     f"Epoch {epoch+1} | Step {global_step:4d} | "
#                     f"Docs {docs_processed:5d} | "
#                     f"Loss {mean_loss.item():.4f} | "
#                     f"Pool {len(all_queries)} queries"
#                 )

#             if docs_processed > 0 and \
#                docs_processed % args.dpr_retrain_every == 0:

#                 logger.info(f"\n{'-'*55}")
#                 logger.info(f"DPR retrain @ {docs_processed} docs "
#                              f"| pool = {len(all_queries)} queries")
#                 logger.info(f"{'-'*55}")

#                 save_weak_queries_jsonl(
#                     all_queries,
#                     os.path.join(args.output_dir,
#                                  f"queries_step{docs_processed}.jsonl")
#                 )
#                 save_weak_qrels_tsv(
#                     all_qrels,
#                     os.path.join(args.output_dir,
#                                  f"qrels_step{docs_processed}.tsv")
#                 )

#                 dpr.retrain(all_queries, all_qrels, args)

#                 ndcg = compute_ndcg_at_10(dpr, eval_pairs)
#                 ndcg_history.append(ndcg)
#                 logger.info(f"NDCG@10 = {ndcg:.4f}  "
#                              f"(best = {best_ndcg:.4f})")

#                 if ndcg > best_ndcg + args.ndcg_min_delta:
#                     best_ndcg     = ndcg
#                     no_improve_ct = 0
#                     _save_best(peft_model, args.output_dir)
#                     args.kl_coeff = max(0.01, args.kl_coeff * 0.95)
#                     logger.info(f"New best - relaxing KL coeff to "
#                                 f"{args.kl_coeff:.4f}")
#                 else:
#                     no_improve_ct += 1
#                     args.kl_coeff = min(0.10, args.kl_coeff * 1.05)
#                     logger.info(f"No improvement "
#                                 f"({no_improve_ct}/{args.patience}). "
#                                 f"KL coeff -> {args.kl_coeff:.4f}")

#                 if no_improve_ct >= args.patience:
#                     logger.info("Early stopping triggered.")
#                     _save_final(peft_model, all_queries, all_qrels,
#                                 ndcg_history, best_ndcg, args)
#                     return

#         avg = float(np.mean(epoch_loss)) if epoch_loss else 0.0
#         logger.info(f"Epoch {epoch+1} done. Avg loss: {avg:.4f}")
#         peft_model.save(os.path.join(args.output_dir, f"peft_epoch{epoch+1}"))

#     _save_final(peft_model, all_queries, all_qrels,
#                 ndcg_history, best_ndcg, args)


# # ─────────────────────────────────────────────────────────────────────────────
# # 10. SAVE HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def _save_best(peft_model: PeftSoftPromptModel, output_dir: str):
#     peft_model.save(os.path.join(output_dir, "best_peft_adapter"))


# def _save_final(peft_model: PeftSoftPromptModel,
#                 all_queries: List[Dict],
#                 all_qrels:   List[Tuple],
#                 ndcg_history: List[float],
#                 best_ndcg:    float,
#                 args):
#     peft_model.save(os.path.join(args.output_dir, "final_peft_adapter"))
#     save_weak_queries_jsonl(
#         all_queries,
#         os.path.join(args.output_dir, "final_weak_queries.jsonl")
#     )
#     save_weak_qrels_tsv(
#         all_qrels,
#         os.path.join(args.output_dir, "final_weak_train.tsv")
#     )
#     with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
#         json.dump({
#             "best_ndcg_at_10":       best_ndcg,
#             "ndcg_history":          ndcg_history,
#             "total_queries_in_pool": len(all_queries),
#             "final_kl_coeff":        args.kl_coeff,
#         }, f, indent=2)

#     dataset_core = args.dataset_name.split("_")[0]
#     logger.info("=" * 60)
#     logger.info("Phase 2 GRPO complete")
#     logger.info(f"  Best NDCG@10        : {best_ndcg:.4f}")
#     logger.info(f"  Total queries pool  : {len(all_queries)}")
#     logger.info(f"  Output dir          : {args.output_dir}")
#     logger.info("")
#     logger.info("Run final DPR training with all accumulated queries:")
#     logger.info(f"  python zhiyuan/dpr_eval.py \\")
#     logger.info(f"    --dataset_name {dataset_core} \\")
#     logger.info(f"    --version v1 --gpu_id 0 \\")
#     logger.info(f"    --train_num {args.train_num} \\")
#     logger.info(f"    --weak_num {len(all_queries)} \\")
#     logger.info(f"    --exp_names grpo_phase2")
#     logger.info("=" * 60)


# if __name__ == "__main__":
#     main()
"""
SPTAR Phase 2: GRPO Soft-Prompt Improvement Loop
=================================================
Fixes applied:
  1. DPR SentenceTransformer forced to CPU (avoids CUDA driver error)
  2. get_ref_log_prob uses CPU tensors to avoid CUDACachingAllocator crash
  3. torch_dtype -> dtype deprecation fixed
  4. ref_encoder_state swap done safely without touching GPU allocator
  5. Phase 1 fiqa numpy weights loaded into prompt encoder
  6. 300 random docs sampled per epoch from full 40k corpus
"""

import os, sys, json, csv, argparse, random, logging, math, warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import PeftConfig

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "xuyang"))
sys.path.insert(0, str(REPO_ROOT / "zhiyuan"))

from default_prompt import DefaultPrompt

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/grpo_phase2_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),          # still prints to terminal too
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("SPTAR Phase 2 - GRPO soft-prompt improvement")

    p.add_argument("--peft_model_id",  required=True)
    p.add_argument("--dpr_ckpt",       required=True)
    p.add_argument("--weak_queries",   required=True)
    p.add_argument("--weak_qrels",     required=True)
    p.add_argument("--corpus",         required=True)
    p.add_argument("--eval_corpus",    required=True)
    p.add_argument("--dev_queries",    required=True)
    p.add_argument("--dev_qrels",      required=True)
    p.add_argument("--output_dir",     default="output/grpo_phase2")

    p.add_argument("--phase1_npy", type=str,
                   default="./xuyang/llm_models/v1_fiqa_50_llama-7b_llama-7b_CAUSAL_LM_TEXT_50_50_3_2023-06-04_0",
                   help="Path to phase1 numpy soft-prompt weights (.npy or raw)")

    p.add_argument("--dataset_name", default="fiqa_50",
                   choices=["fiqa_50", "ms_50", "hotpotqa_50", "fever_50"])
    p.add_argument("--prompt_num",   type=int, default=2)
    p.add_argument("--text_len",     type=int, default=350)
    p.add_argument("--train_num",    type=int, default=50)

    p.add_argument("--docs_per_epoch", type=int, default=300,
                   help="Number of docs randomly sampled per epoch from full corpus")

    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--base_model",   type=str, default=None)

    p.add_argument("--grpo_group_size", type=int,   default=4)
    p.add_argument("--lr",              type=float, default=3e-2)
    p.add_argument("--grpo_epochs",     type=int,   default=3)
    p.add_argument("--batch_docs",      type=int,   default=1)
    p.add_argument("--kl_coeff",        type=float, default=0.04)
    p.add_argument("--clip_ratio",      type=float, default=0.2)

    p.add_argument("--w_relevance",      type=float, default=0.4)
    p.add_argument("--w_retrievability", type=float, default=0.4)
    p.add_argument("--w_specificity",    type=float, default=0.2)

    p.add_argument("--dpr_retrain_every",  type=int, default=200)
    p.add_argument("--dpr_retrain_epochs", type=int, default=2)
    p.add_argument("--dpr_batch_size",     type=int, default=16)

    p.add_argument("--patience",       type=int,   default=4)
    p.add_argument("--ndcg_min_delta", type=float, default=0.001)

    p.add_argument("--device",         type=str,   default="cuda:0")
    p.add_argument("--max_new_tokens", type=int,   default=64)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--seed",           type=int,   default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_weak_queries_jsonl(queries: List[Dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for q in queries:
            json.dump(q, f)
            f.write("\n")


def save_weak_qrels_tsv(qrels: List[Tuple], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query-id", "corpus-id", "score"])
        for row in qrels:
            w.writerow(row)


def load_corpus(path: str) -> List[Dict]:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        corpus = []
        for _, row in df.iterrows():
            corpus.append({
                "_id":   str(row["_id"]),
                "title": str(row.get("title", "")),
                "text":  str(row["text"]),
            })
    else:
        corpus = load_jsonl(path)
    logger.info(f"Corpus: {len(corpus)} documents from {path}")
    return corpus


def load_phase1_weak_data(queries_path: str, qrels_path: str
                          ) -> Tuple[List[Dict], List[Tuple]]:
    queries = load_jsonl(queries_path)
    qrels: List[Tuple] = []
    with open(qrels_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qrels.append((row["query-id"], row["corpus-id"], row["score"]))
    logger.info(f"Phase 1 data: {len(queries)} queries, {len(qrels)} qrels")
    return queries, qrels


def cut_text(text: str, max_len: int = 350) -> str:
    words = text.split()
    return " ".join(words[:max_len]) if len(words) > max_len else text


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def get_fixed_prompts(dataset_name: str) -> Tuple[str, str]:
    if "ms_" in dataset_name:
        return (DefaultPrompt.ms_50_fixed_one_shot_prompt,
                DefaultPrompt.ms_50_fixed_two_shot_prompt)
    if dataset_name == "fiqa_50":
        return (DefaultPrompt.fiqa_50_fixed_one_shot_prompt,
                DefaultPrompt.fiqa_50_fixed_two_shot_prompt)
    if dataset_name == "hotpotqa_50":
        return (DefaultPrompt.hotpotqa_50_fixed_one_shot_prompt,
                DefaultPrompt.hotpotqa_50_fixed_two_shot_prompt)
    if dataset_name == "fever_50":
        return (DefaultPrompt.fever_50_fixed_one_shot_prompt,
                DefaultPrompt.fever_50_fixed_two_shot_prompt)
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def build_prompt(corpus_text: str, prompt_num: int,
                 one_shot: str, two_shot: str,
                 text_len: int = 350) -> str:
    corpus_text = cut_text(corpus_text, text_len)
    if prompt_num == 2:
        return "{} \n Document: {} \n Relevant Query: ".format(one_shot, corpus_text)
    elif prompt_num == 3:
        return "{} \n Document: {} \n Relevant Query: ".format(two_shot, corpus_text)
    else:
        return "Document: {} \n Relevant Query: ".format(corpus_text)


def simple_filter(text: str) -> str:
    text = text.split("\n")[0]
    for punt in [".", ",", "?"]:
        pre_i, new_text = "", ""
        for i in text.split(punt):
            if pre_i != i:
                new_text += i
                pre_i = i
            else:
                break
        text = new_text
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. PEFT MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class PeftSoftPromptModel:

    def __init__(self, peft_model_id: str, device: torch.device,
                 load_in_4bit: bool = True,
                 base_model_override: str = None,
                 phase1_npy: str = None):
        self.device = device

        logger.info(f"Loading PEFT config from {peft_model_id}")
        peft_model_id = str(Path(peft_model_id).resolve())

        config    = PeftConfig.from_pretrained(peft_model_id)
        base_path = config.base_model_name_or_path

        if base_model_override:
            base_path = base_model_override
            logger.info(f"Base model overridden to: {base_path}")
        elif not os.path.exists(base_path):
            logger.warning(f"base_model_name_or_path {base_path!r} not found locally.")
            logger.warning("Falling back to meta-llama/Llama-2-7b-hf from HF Hub.")
            logger.warning("Set --base_model to override if you have a local copy.")
            base_path = "meta-llama/Llama-2-7b-hf"
        logger.info(f"Base LLM: {base_path}")

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_path,
                quantization_config=bnb_config,
                device_map={"": 0},
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            logger.info("Base LLM loaded in 4-bit NF4")
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_path,
                dtype=torch.float16,
                device_map={"": device},
            )

        from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

        num_virtual_tokens = getattr(config, "num_virtual_tokens", 50)
        init_text = getattr(config, "prompt_tuning_init_text",
                            "please generate query for this document")

        logger.info(f"Creating PEFT adapter: "
                    f"num_virtual_tokens={num_virtual_tokens}, "
                    f"init_text='{init_text}'")

        peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=init_text,
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=base_path,
        )
        self.model = get_peft_model(base, peft_config)

        # ── Load Phase 1 fiqa numpy weights into prompt encoder ──────────────
        if phase1_npy and os.path.exists(phase1_npy):
            arr    = np.load(phase1_npy)
            tensor = torch.tensor(arr, dtype=torch.float32)
            loaded = False
            for name, param in self.model.prompt_encoder.named_parameters():
                if param.shape == tensor.shape:
                    param.data = tensor
                    loaded = True
                    logger.info(f"Loaded Phase 1 fiqa weights "
                                f"({arr.shape}) into prompt_encoder.{name}")
                    break
            if not loaded:
                logger.warning(
                    f"Phase 1 npy shape {arr.shape} did not match any "
                    f"prompt_encoder param — starting from random init."
                )
        else:
            logger.warning(
                f"phase1_npy not found at {phase1_npy!r} — "
                f"starting from random init (queries may degrade)."
            )
        # ─────────────────────────────────────────────────────────────────────

        for name, param in self.model.named_parameters():
            if "prompt_encoder" in name:
                param.requires_grad = True
                param.data = param.data.float()
            else:
                param.requires_grad = False

        n_train = sum(p.numel() for p in self.model.parameters()
                      if p.requires_grad)
        logger.info(f"Trainable params: {n_train:,} (prompt encoder only)")

        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def save(self, path: str):
        self.model.save_pretrained(path)
        logger.info(f"Saved PEFT adapter -> {path}")

    @torch.no_grad()
    def generate_query(self, prompt: str,
                       max_new_tokens: int = 64,
                       temperature: float = 0.7) -> str:
        device = self.device

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
        )

        # 🔥 Move EVERYTHING to GPU (fixes your error)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if hasattr(self.model, "prompt_encoder"):
            self.model.prompt_encoder.to(device)
        

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                do_sample=True,
            )
        full_text = self.tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0]
        query = full_text[len(prompt):]
        del outputs
        torch.cuda.empty_cache()
        return simple_filter(query)

    @torch.no_grad()
    def generate_group(self, prompt: str, G: int,
                       max_new_tokens: int,
                       temperature: float) -> List[str]:
        results = []
        for _ in range(G):
            q = self.generate_query(prompt, max_new_tokens, temperature)
            results.append(q)
        return results

    def compute_log_prob(self, prompt: str, query: str) -> torch.Tensor:
        torch.cuda.empty_cache()

        full_text  = prompt + query
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
        ).input_ids.to(self.device)
        full_ids = self.tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=576
        ).input_ids.to(self.device)

        answer_len = full_ids.shape[1] - prompt_ids.shape[1]
        if answer_len <= 0:
            return sum(p.sum() * 0.0
                       for p in self.model.parameters()
                       if p.requires_grad)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = self.model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
            )

        prompt_len = prompt_ids.shape[1]
        logits     = out.logits[0].float()
        pred       = logits[prompt_len - 1: prompt_len - 1 + answer_len]
        targets    = full_ids[0, prompt_len: prompt_len + answer_len]

        n         = min(pred.shape[0], targets.shape[0])
        log_probs = F.log_softmax(pred[:n], dim=-1)
        result    = log_probs[
            torch.arange(n, device=self.device), targets[:n]
        ].sum()

        del out, logits, pred, log_probs
        torch.cuda.empty_cache()

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. DPR MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class DPRManager:

    def __init__(self, dpr_ckpt: str, corpus: List[Dict],
                 device: torch.device):
        self.device = device
        self.corpus = corpus
        self._load_model(dpr_ckpt)
        self._build_index()

    def _load_model(self, path: str):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(path, device="cpu")
            logger.info(f"DPR loaded from {path} (on CPU)")
        except Exception as e:
            logger.warning(f"DPR load failed ({e}). "
                           "Retrievability reward will return 0.5.")
            self.model = None

    def _build_index(self):
        if self.model is None:
            self.corpus_embeddings = None
            self.corpus_ids        = []
            return
        logger.info("Building DPR dense index ...")
        self.corpus_ids   = [d["_id"]  for d in self.corpus]
        corpus_texts      = [d["text"] for d in self.corpus]
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda:0",
            normalize_embeddings=True,
        ).cpu()
        logger.info(f"DPR index shape: {self.corpus_embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        if self.model is None or self.corpus_embeddings is None:
            return []
        q_emb = self.model.encode(
            query, convert_to_tensor=True,
            device="cpu", normalize_embeddings=True
        )
        scores  = torch.matmul(self.corpus_embeddings, q_emb)
        top_idx = scores.topk(
            min(top_k, len(self.corpus_ids))
        ).indices.tolist()
        return [self.corpus_ids[i] for i in top_idx]

    def retrain(self, all_queries: List[Dict], all_qrels: List[Tuple],
                args) -> None:
        import subprocess, shutil

        dataset_core  = args.dataset_name.split("_")[0]
        weak_num_str  = str(len(all_queries))
        exp_tag       = "grpo_phase2"
        xuyang_dir    = str(REPO_ROOT / "xuyang" / "data")
        target_dir    = os.path.join(
            xuyang_dir, args.dataset_name, weak_num_str
        )
        os.makedirs(target_dir, exist_ok=True)

        wq_path = os.path.join(
            target_dir,
            f"weak_queries_{args.train_num}_{exp_tag}.jsonl"
        )
        wt_path = os.path.join(
            target_dir,
            f"weak_train_{args.train_num}_{exp_tag}.tsv"
        )
        save_weak_queries_jsonl(all_queries, wq_path)
        save_weak_qrels_tsv(all_qrels, wt_path)
        logger.info(f"Wrote {len(all_queries)} queries -> {wq_path}")

        beir_dir = os.path.join(
            str(REPO_ROOT / "zhiyuan"),
            "datasets", "raw", "beir", dataset_core
        )
        eval_corpus_dest = os.path.join(
            beir_dir,
            f"corpus_{weak_num_str}_reduced_ratio_20.jsonl"
        )
        if not os.path.exists(eval_corpus_dest):
            shutil.copy(args.eval_corpus, eval_corpus_dest)
            logger.info(f"Copied eval corpus -> {eval_corpus_dest}")

        train_script = str(
            REPO_ROOT / "zhiyuan" / "retriever" / "dpr"
            / "train" / "train_sbert.py"
        )
        cmd = [
            sys.executable, train_script,
            "--dataset_name", dataset_core,
            "--num_epochs",   str(args.dpr_retrain_epochs),
            "--train_num",    str(args.train_num),
            "--weak_num",     weak_num_str,
            "--exp_name",     exp_tag,
        ]
        logger.info(f"DPR retrain command:\n  {' '.join(cmd)}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            logger.error("DPR retrain subprocess failed - keeping old model.")
            return

        model_name   = "bert-large-uncased"
        new_dpr_path = os.path.join(
            str(REPO_ROOT / "zhiyuan"),
            "retriever", "dpr", "train", "output",
            exp_tag, str(args.train_num),
            f"{model_name}-v1-{dataset_core}"
        )
        if os.path.exists(new_dpr_path):
            self._load_model(new_dpr_path)
            self._build_index()
            logger.info(f"DPR reloaded from {new_dpr_path}")
        else:
            logger.warning(
                f"DPR output path not found: {new_dpr_path}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. REWARD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def reward_relevance(query: str, doc_text: str) -> float:
    if not query.strip():
        return 0.0
    q_toks = set(query.lower().split())
    d_toks = set(doc_text.lower().split())
    if not q_toks:
        return 0.0
    overlap      = len(q_toks & d_toks) / len(q_toks)
    length_bonus = min(1.0, len(query.split()) / 8.0)
    return 0.6 * overlap + 0.4 * length_bonus


def reward_specificity(query: str) -> float:
    q = query.lower().strip()
    if not q:
        return 0.0

    # Hard reject non-ASCII heavy or punctuation heavy queries
    import re
    non_alpha = sum(1 for c in q if not c.isalpha() and c != ' ')
    if non_alpha / max(len(q), 1) > 0.15:
        return 0.0

    score = 1.0
    if any(q.startswith(g) for g in
           ["is ", "are ", "was ", "were ", "do ", "does ",
            "did ", "can ", "could ", "would ", "should "]):
        score *= 0.6
    if any(q.startswith(w) for w in
           ["what", "who", "when", "where", "why", "how", "which"]):
        score = min(1.0, score * 1.3)
    n = len(q.split())
    if n < 4:
        score *= 0.3
    elif n > 25:
        score *= 0.7
    return min(1.0, score)


def reward_retrievability(query: str, corpus_id: str,
                          dpr: DPRManager, top_k: int = 10) -> float:
    if dpr.model is None:
        return 0.5
    results = dpr.retrieve(query, top_k=top_k)
    if corpus_id in results:
        rank = results.index(corpus_id) + 1
        return 1.0 / math.log2(rank + 1)
    return 0.0


def composite_reward(query: str, doc_text: str, corpus_id: str,
                     dpr: DPRManager, args) -> float:
    q = query.strip()
    words = q.split()

    # Hard reject degenerate outputs
    if len(words) < 3:
        return 0.0

    # Penalize junk word heavy queries
    junk_words = {
        "nobody", "everybody", "hopefully", "obviously",
        "surely", "anybody", "someone", "ultimately",
        "the", "a", "in", "of", "and", "or", "but"
    }
    junk_ratio = sum(
        1 for w in words if w.lower() in junk_words
    ) / len(words)
    if junk_ratio > 0.5:
        return 0.0

    return (args.w_relevance      * reward_relevance(q, doc_text)
          + args.w_retrievability * reward_retrievability(
                q, corpus_id, dpr)
          + args.w_specificity    * reward_specificity(q))


# ─────────────────────────────────────────────────────────────────────────────
# 7. NDCG@10
# ─────────────────────────────────────────────────────────────────────────────

def compute_ndcg_at_10(dpr: DPRManager,
                       eval_pairs: List[Tuple[str, str]],
                       top_k: int = 10) -> float:
    if dpr.model is None or not eval_pairs:
        return 0.0
    scores = []
    for query_text, corpus_id in eval_pairs:
        results = dpr.retrieve(query_text, top_k=top_k)
        if corpus_id in results:
            rank = results.index(corpus_id) + 1
            scores.append(1.0 / math.log2(rank + 1))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# 8. GRPO LOSS
# ─────────────────────────────────────────────────────────────────────────────

def grpo_loss(curr_log_probs: List[torch.Tensor],
              ref_log_probs:  List[float],
              rewards:        List[float],
              kl_coeff:       float,
              clip_ratio:     float) -> torch.Tensor:
    if not rewards:
        return torch.tensor(0.0, requires_grad=True)

    r_t   = torch.tensor(rewards, dtype=torch.float32)
    adv_t = (r_t - r_t.mean()) / (r_t.std() + 1e-8)

    losses = []
    for lp_curr, lp_ref, adv in zip(curr_log_probs, ref_log_probs, adv_t):
        adv      = adv.to(lp_curr.device)
        ratio    = torch.exp(lp_curr - float(lp_ref))
        clipped  = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pol_loss = -torch.min(ratio * adv, clipped * adv)
        kl_pen   = kl_coeff * (lp_curr - float(lp_ref))
        losses.append(pol_loss + kl_pen)

    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load full corpus (used for DPR index + epoch sampling)
    full_corpus                  = load_corpus(args.corpus)
    phase1_queries, phase1_qrels = load_phase1_weak_data(
        args.weak_queries, args.weak_qrels
    )

    random.shuffle(phase1_queries)
    split        = max(1, int(0.8 * len(phase1_queries)))
    seed_queries = phase1_queries[:split]
    eval_queries = phase1_queries[split:]
    seed_ids     = {q["_id"] for q in seed_queries}
    seed_qrels   = [qr for qr in phase1_qrels if qr[0] in seed_ids]

    qid_to_cid  = {qr[0]: qr[1] for qr in phase1_qrels}
    eval_pairs: List[Tuple[str, str]] = [
        (q["text"], qid_to_cid[q["_id"]])
        for q in eval_queries
        if q["_id"] in qid_to_cid
    ]
    logger.info(f"Seed: {len(seed_queries)} queries | "
                f"Eval: {len(eval_queries)} queries | "
                f"Eval pairs: {len(eval_pairs)}")

    one_shot, two_shot = get_fixed_prompts(args.dataset_name)

    peft_model = PeftSoftPromptModel(
        peft_model_id=args.peft_model_id,
        device=device,
        load_in_4bit=args.load_in_4bit,
        base_model_override=args.base_model,
        phase1_npy=args.phase1_npy,
    )

    # Snapshot reference encoder state (after phase1 weights are loaded)
    ref_encoder_state = {
        k: v.detach().cpu().clone().float()
        for k, v in peft_model.model.prompt_encoder.state_dict().items()
    }
    logger.info("Reference encoder state snapshotted from Phase 1 weights")

    def get_ref_log_prob(prompt: str, query: str) -> float:
        live_state = {
            k: v.detach().cpu().clone()
            for k, v in peft_model.model.prompt_encoder.state_dict().items()
        }
        enc_device = next(
            peft_model.model.prompt_encoder.parameters()
        ).device
        peft_model.model.prompt_encoder.load_state_dict(
            {k: v.to(enc_device) for k, v in ref_encoder_state.items()}
        )
        with torch.no_grad():
            lp = peft_model.compute_log_prob(prompt, query).item()
        peft_model.model.prompt_encoder.load_state_dict(
            {k: v.to(enc_device) for k, v in live_state.items()}
        )
        return lp

    # DPR loads on full corpus for meaningful retrieval evaluation
    dpr = DPRManager(args.dpr_ckpt, full_corpus, device)

    baseline_ndcg = compute_ndcg_at_10(dpr, eval_pairs)
    logger.info(f"Baseline NDCG@10 = {baseline_ndcg:.4f}")
    ndcg_history  = [baseline_ndcg]
    best_ndcg     = baseline_ndcg
    no_improve_ct = 0

    trainable   = [p for p in peft_model.model.parameters()
                   if p.requires_grad]
    optimizer   = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = (args.docs_per_epoch // max(1, args.batch_docs)) \
                  * args.grpo_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, total_steps // 20),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")

    all_queries: List[Dict]  = list(seed_queries)
    all_qrels:   List[Tuple] = list(seed_qrels)
    next_qid = 3_000_000

    docs_processed = 0
    global_step    = 0

    logger.info("=" * 60)
    logger.info("Phase 2 GRPO loop starting")
    logger.info(f"  Full corpus:         {len(full_corpus)} docs")
    logger.info(f"  Docs per epoch:      {args.docs_per_epoch} (random sample)")
    logger.info(f"  Group size G:        {args.grpo_group_size}")
    logger.info(f"  Docs per step:       {args.batch_docs}")
    logger.info(f"  Max new tokens:      {args.max_new_tokens}")
    logger.info(f"  DPR retrain every:   {args.dpr_retrain_every} docs")
    logger.info(f"  Prompt format:       prompt_num={args.prompt_num}")
    logger.info("=" * 60)

    for epoch in range(args.grpo_epochs):

        # ── Sample 300 random docs fresh each epoch ───────────────────────
        epoch_corpus = random.sample(
            full_corpus, min(args.docs_per_epoch, len(full_corpus))
        )
        logger.info(f"Epoch {epoch+1}: sampled {len(epoch_corpus)} docs "
                    f"from {len(full_corpus)} total")
        # ──────────────────────────────────────────────────────────────────

        batches    = [
            epoch_corpus[i: i + args.batch_docs]
            for i in range(0, len(epoch_corpus), args.batch_docs)
        ]
        epoch_loss = []

        for batch in batches:
            optimizer.zero_grad()
            batch_loss = None

            for doc in batch:
                corpus_id = doc["_id"]
                doc_text  = doc["text"]

                # Skip numeric/junk documents
                words = doc_text.split()
                alpha_words = [w for w in words if any(c.isalpha() for c in w)]
                if len(alpha_words) < 10 or len(alpha_words) / max(len(words), 1) < 0.5:
                    docs_processed += 1
                    continue
                logger.info(f"  [DOC {docs_processed+1}] Doc text preview: {doc_text[:100]}")

                prompt    = build_prompt(doc_text, args.prompt_num,
                                         one_shot, two_shot, args.text_len)

                import time
                logger.info(
                    f"  [DOC {docs_processed+1}] Starting generation..."
                )
                t0 = time.time()
                queries_g = peft_model.generate_group(
                    prompt, args.grpo_group_size,
                    args.max_new_tokens, args.temperature
                )
                torch.cuda.empty_cache()
                logger.info(
                    f"  [DOC {docs_processed+1}] Generation done in "
                    f"{time.time()-t0:.1f}s | queries: {queries_g}"
                )

                queries_g = [q for q in queries_g if q.strip()]
                if not queries_g:
                    logger.info(
                        f"  [DOC {docs_processed+1}] All queries empty, skipping"
                    )
                    continue

                t1 = time.time()
                rewards_g = [
                    composite_reward(q, doc_text, corpus_id, dpr, args)
                    for q in queries_g
                ]
                logger.info(
                    f"  [DOC {docs_processed+1}] Rewards done in "
                    f"{time.time()-t1:.1f}s | rewards: {rewards_g}"
                )

                t2 = time.time()
                ref_lps = [get_ref_log_prob(prompt, q) for q in queries_g]
                logger.info(
                    f"  [DOC {docs_processed+1}] Ref log probs done in "
                    f"{time.time()-t2:.1f}s"
                )

                t3 = time.time()
                curr_lps = [
                    peft_model.compute_log_prob(prompt, q)
                    for q in queries_g
                ]
                logger.info(
                    f"  [DOC {docs_processed+1}] Curr log probs done in "
                    f"{time.time()-t3:.1f}s"
                )

                doc_loss   = grpo_loss(curr_lps, ref_lps, rewards_g,
                                       args.kl_coeff, args.clip_ratio)
                batch_loss = doc_loss if batch_loss is None \
                             else batch_loss + doc_loss

                threshold = float(np.mean(rewards_g))
                for q, r in zip(queries_g, rewards_g):
                    if r >= threshold:
                        next_qid += 1
                        all_queries.append(
                            {"_id": str(next_qid), "text": q,
                             "metadata": {}}
                        )
                        all_qrels.append((str(next_qid), corpus_id, "1"))

            if batch_loss is None:
                continue

            mean_loss = batch_loss / len(batch)
            scaler.scale(mean_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            torch.cuda.empty_cache()

            epoch_loss.append(mean_loss.item())
            docs_processed += len(batch)
            global_step    += 1

            if global_step % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1} | Step {global_step:4d} | "
                    f"Docs {docs_processed:5d} | "
                    f"Loss {mean_loss.item():.4f} | "
                    f"Pool {len(all_queries)} queries"
                )

            if docs_processed > 0 and \
               docs_processed % args.dpr_retrain_every == 0:

                logger.info(f"\n{'-'*55}")
                logger.info(f"DPR retrain @ {docs_processed} docs "
                             f"| pool = {len(all_queries)} queries")
                logger.info(f"{'-'*55}")

                save_weak_queries_jsonl(
                    all_queries,
                    os.path.join(args.output_dir,
                                 f"queries_step{docs_processed}.jsonl")
                )
                save_weak_qrels_tsv(
                    all_qrels,
                    os.path.join(args.output_dir,
                                 f"qrels_step{docs_processed}.tsv")
                )

                dpr.retrain(all_queries, all_qrels, args)

                ndcg = compute_ndcg_at_10(dpr, eval_pairs)
                ndcg_history.append(ndcg)
                logger.info(f"NDCG@10 = {ndcg:.4f}  "
                             f"(best = {best_ndcg:.4f})")

                if ndcg > best_ndcg + args.ndcg_min_delta:
                    best_ndcg     = ndcg
                    no_improve_ct = 0
                    _save_best(peft_model, args.output_dir)
                    args.kl_coeff = max(0.01, args.kl_coeff * 0.95)
                    logger.info(f"New best - relaxing KL coeff to "
                                f"{args.kl_coeff:.4f}")
                else:
                    no_improve_ct += 1
                    args.kl_coeff = min(0.10, args.kl_coeff * 1.05)
                    logger.info(f"No improvement "
                                f"({no_improve_ct}/{args.patience}). "
                                f"KL coeff -> {args.kl_coeff:.4f}")

                if no_improve_ct >= args.patience:
                    logger.info("Early stopping triggered.")
                    _save_final(peft_model, all_queries, all_qrels,
                                ndcg_history, best_ndcg, args)
                    return

        avg = float(np.mean(epoch_loss)) if epoch_loss else 0.0
        logger.info(f"Epoch {epoch+1} done. Avg loss: {avg:.4f}")
        peft_model.save(
            os.path.join(args.output_dir, f"peft_epoch{epoch+1}")
        )

    _save_final(peft_model, all_queries, all_qrels,
                ndcg_history, best_ndcg, args)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_best(peft_model: PeftSoftPromptModel, output_dir: str):
    peft_model.save(os.path.join(output_dir, "best_peft_adapter"))


def _save_final(peft_model: PeftSoftPromptModel,
                all_queries: List[Dict],
                all_qrels:   List[Tuple],
                ndcg_history: List[float],
                best_ndcg:    float,
                args):
    peft_model.save(os.path.join(args.output_dir, "final_peft_adapter"))
    save_weak_queries_jsonl(
        all_queries,
        os.path.join(args.output_dir, "final_weak_queries.jsonl")
    )
    save_weak_qrels_tsv(
        all_qrels,
        os.path.join(args.output_dir, "final_weak_train.tsv")
    )
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({
            "best_ndcg_at_10":       best_ndcg,
            "ndcg_history":          ndcg_history,
            "total_queries_in_pool": len(all_queries),
            "final_kl_coeff":        args.kl_coeff,
        }, f, indent=2)

    dataset_core = args.dataset_name.split("_")[0]
    logger.info("=" * 60)
    logger.info("Phase 2 GRPO complete")
    logger.info(f"  Best NDCG@10        : {best_ndcg:.4f}")
    logger.info(f"  Total queries pool  : {len(all_queries)}")
    logger.info(f"  Output dir          : {args.output_dir}")
    logger.info("")
    logger.info("Run final DPR training with all accumulated queries:")
    logger.info(f"  python zhiyuan/dpr_eval.py \\")
    logger.info(f"    --dataset_name {dataset_core} \\")
    logger.info(f"    --version v1 --gpu_id 0 \\")
    logger.info(f"    --train_num {args.train_num} \\")
    logger.info(f"    --weak_num {len(all_queries)} \\")
    logger.info(f"    --exp_names grpo_phase2")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

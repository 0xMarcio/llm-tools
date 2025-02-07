#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # Hide Python and PyTorch warnings
import os
import requests
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from bs4 import BeautifulSoup
import tempfile
import readline

"""
--------------------------------------------------------------------------------
FROM-SCRATCH TRANSFORMER WITH A CUSTOM BPE TOKENIZER (NO PRETRAINED MODEL),
FALLBACK IF OUTPUT IS GIBBERISH, AND ONLY USE AUTOCAST ON CUDA (AVOID MPS ISSUES)

1) We'll scrape Hacker News (up to 5 pages) until we get 50 valid links (≥500 chars).
2) Train a custom BPE tokenizer with no progress bars or warnings.
3) Build a small GPT-like model, training in float32 on CPU/MPS, or with AMP on CUDA.
4) After generation, we check if there's at least one period. If not, we replace
   the text with a fallback sentence. This ensures there's always at least one
   coherent line.

Note:
 - If you're on Apple MPS, you'll just get normal float32. 
 - If you're on CUDA, we do `torch.cuda.amp.autocast` (the old signature),
   which actually works without device_type for most versions of PyTorch.
 - This avoids all "device_type" parameter issues and warning spam.

Yes, output is still mostly nonsense because the model is too small and
dataset is tiny. The fallback ensures at least a coherent line.
--------------------------------------------------------------------------------
"""

HN_BASE_URL = "https://news.ycombinator.com/"
UNWANTED_DOMAINS = ["youtube.com", "youtu.be"]
DESIRED_PAGES = 50
MAX_HN_PAGES = 5
MIN_CHARS = 500

VOCAB_SIZE = 2000
MAX_SEQ_LEN = 256
BATCH_SIZE = 16
TRAINING_STEPS = 5000
LEARNING_RATE = 3e-4

N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
TEMPERATURE = 0.7
TOP_K = 10

FALLBACK_SENTENCE = "We could not come up with anything clever to say."

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
except ImportError:
    print("[ERROR] Install 'tokenizers' first: pip install tokenizers")
    exit()

def fetch_hn_page(page_num=1):
    url = f"{HN_BASE_URL}news?p={page_num}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title_tags = soup.select(".athing .titleline a")
    return [(t.get_text(" ", strip=True), t.get("href", "")) for t in title_tags]

def is_valid_link(link: str) -> bool:
    if not link.lower().startswith("http"):
        return False
    for dom in UNWANTED_DOMAINS:
        if dom in link.lower():
            return False
    return True

def fetch_body_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        if not soup.body:
            return ""
        return soup.body.get_text(separator=" ", strip=True)
    except:
        return ""

def gather_data():
    valid_data = []
    page_num = 1
    while len(valid_data) < DESIRED_PAGES and page_num <= MAX_HN_PAGES:
        items = fetch_hn_page(page_num)
        page_num += 1
        for (title, link) in items:
            if len(valid_data) >= DESIRED_PAGES:
                break
            if not is_valid_link(link):
                continue
            txt = fetch_body_text(link)
            if len(txt) < MIN_CHARS:
                continue
            valid_data.append((title, link, txt))
    return valid_data

print(f">> Gathering 50 pages from Hacker News ...\n")
pages = gather_data()
if not pages:
    print(">> No valid pages found.")
    exit()

all_texts = [p[2] for p in pages]
combined_corpus = "\n".join(all_texts).strip()
if not combined_corpus:
    print(">> Combined corpus empty.")
    exit()

print(f">> Building BPE tokenizer (vocab_size={VOCAB_SIZE}), no progress bars...\n")
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=False,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"],
    )
    corpus_file = os.path.join(tmpdir, "corpus.txt")
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write(combined_corpus)
    tokenizer.train([corpus_file], trainer)

def tokenize(text: str):
    return tokenizer.encode(text).ids

def detokenize(ids: list):
    return tokenizer.decode(ids)

vocab_size = tokenizer.get_vocab_size()

all_ids = tokenize(combined_corpus)
num_chunks = len(all_ids) // MAX_SEQ_LEN
if num_chunks < 1:
    print(f">> Not enough tokens to form even 1 chunk.\n")
    exit()

train_data = [all_ids[i * MAX_SEQ_LEN : (i + 1) * MAX_SEQ_LEN] for i in range(num_chunks)]
train_data = torch.tensor(train_data, dtype=torch.long)

def get_batch(batch_size=BATCH_SIZE):
    idx = torch.randint(0, train_data.size(0), (batch_size,))
    x = train_data[idx, :-1]
    y = train_data[idx, 1:]
    return x, y

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        mask = self.mask[:T, :T]
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=50, temperature=0.7, top_k=10):
        for _ in range(max_new_tokens):
            if idx.size(1) > self.block_size:
                idx = idx[:, -self.block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # top-k
            top_vals, top_indices = torch.topk(probs, k=top_k, dim=-1)
            top_probs = top_vals / top_vals.sum(dim=-1, keepdim=True)
            next_ids = []
            for b in range(probs.size(0)):
                cat_idx = torch.multinomial(top_probs[b], 1)
                token_id = top_indices[b, cat_idx]
                next_ids.append(token_id)
            next_ids = torch.stack(next_ids, dim=0)
            next_ids = torch.clamp(next_ids, 0, self.vocab_size - 1)
            idx = torch.cat((idx, next_ids), dim=1)
        return idx

device = "cuda" if torch.cuda.is_available() else ("cpu")  # skip MPS autocast for simplicity
model = TinyGPT(vocab_size, N_EMBD, N_HEAD, N_LAYER, MAX_SEQ_LEN).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

use_amp = (device == "cuda")  # only do autocast on CUDA, skip MPS/CPU
if use_amp:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler(enabled=True)
else:
    # dummy context
    from contextlib import contextmanager
    @contextmanager
    def autocast(enabled=True):
        yield
    class _NoScaler:
        def scale(self, loss):
            return loss
        def step(self, opt):
            opt.step()
        def update(self):
            pass
    scaler = _NoScaler()

print(f">> Training TinyGPT from scratch ({TRAINING_STEPS} steps) on device: {device}...\n")
model.train()
for step in range(TRAINING_STEPS):
    xb, yb = get_batch()
    xb, yb = xb.to(device), yb.to(device)
    with autocast(enabled=use_amp):
        _, loss = model(xb, yb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if step % 200 == 0:
        print(f" Step {step}/{TRAINING_STEPS}, loss={loss.item():.4f}\n")

model.eval()
print(f">> Training done. Type 'quit' or 'exit' to stop.\n")

def fallback_if_nonsense(txt: str) -> str:
    """If there's no '.' or length < 5, fallback to a default sentence."""
    if len(txt) < 5 or '.' not in txt:
        return FALLBACK_SENTENCE
    return txt

while True:
    user_in = input("User: ")
    if user_in.strip().lower() in ["quit", "exit"]:
        break
    user_ids = tokenize(user_in)
    if not user_ids:
        print("\n[Model] (Your input couldn't be tokenized.)\n")
        continue
    ctx = torch.tensor([user_ids], dtype=torch.long, device=device)
    out_ids = model.generate(ctx, max_new_tokens=100, temperature=TEMPERATURE, top_k=TOP_K)[0].tolist()
    new_tokens = out_ids[len(user_ids):]
    new_tokens = [max(0, min(t, vocab_size - 1)) for t in new_tokens]
    raw_text = detokenize(new_tokens).strip()
    raw_text = re.sub(r"\s+", " ", raw_text)
    final_text = fallback_if_nonsense(raw_text)
    print(f"[Model] {final_text}\n")

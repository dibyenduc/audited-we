import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

MODEL, TOK = "roneneldan/TinyStories-33M", "EleutherAI/gpt-neo-125M"

def load_lm(top_k=50):
    tok = AutoTokenizer.from_pretrained(TOK)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval().to("cpu")
    cache = {}

    @torch.no_grad()
    def next_topk(ids):
        key = tuple(ids)
        if key not in cache:
            x = torch.tensor([list(ids)], dtype=torch.long)
            probs = torch.softmax(model(x).logits[0, -1].double(), dim=-1).numpy()
            top = np.sort(np.argsort(-probs, kind="stable")[:top_k])
            q = probs[top] / probs[top].sum()
            cache[key] = (top, q)
        return cache[key]

    return tok, next_topk


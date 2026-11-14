import hashlib, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from toy_we import step_rng

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

MODEL, TOK = "roneneldan/TinyStories-33M", "EleutherAI/gpt-neo-125M"
tok = AutoTokenizer.from_pretrained(TOK)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval().to("cpu")

@torch.no_grad()
def next_probs(ids):
    x = torch.tensor([ids], dtype=torch.long)
    return torch.softmax(model(x).logits[0, -1].double(), dim=-1).numpy()

def sample_token(probs, u):
    c = np.cumsum(probs); c = c / c[-1]
    return min(int(np.searchsorted(c, u, side="right")), len(c) - 1)

def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

prompt = "Once upon a time there was"
ids = tok.encode(prompt)
print("torch", torch.__version__, "| threads", torch.get_num_threads(), "| prompt tokens", len(ids))

p1, p2 = next_probs(ids), next_probs(ids)
print("TEST 1 same process, same input, bitwise equal:", p1.tobytes() == p2.tobytes())
print("probs digest (compare across separate runs):", digest(p1))

seq, root = list(ids), b"det-test"
t0 = time.time()
for t in range(1, 41):
    u = step_rng(root, t, "move").random(1)[0]
    seq.append(sample_token(next_probs(seq), u))
dt = time.time() - t0
print("TEST 2 sampled text:", tok.decode(seq))
print("token sequence digest (compare across separate runs):", digest(np.array(seq)))
print(f"TEST 3 speed: {dt/40*1000:.1f} ms per token (batch size 1)")

a = ids
b = ids[:-1] + [ids[-1] + 1]
with torch.no_grad():
    lb = model(torch.tensor([a, b])).logits[:, -1].double()
pb = torch.softmax(lb, dim=-1).numpy()
pa = next_probs(a)
print("TEST 4 batch vs single, bitwise equal:", pb[0].tobytes() == pa.tobytes(),
      "| max abs diff:", float(np.abs(pb[0] - pa).max()))


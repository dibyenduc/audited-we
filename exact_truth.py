import json, time
from lm import load_lm
from events import exact_event_prob

PROMPT = "Once upon a time there was a little girl named Lily. One day, she went to the park and saw a"
WORDS = ["dragon", "monster", "storm", "fire", "dark", "cry", "angry", "scared",
         "danger", "lost", "fell", "hurt", "loud", "sad", "scary", "bad"]
H, C, TOP_K = 3, 2, 50

tok, next_topk = load_lm(TOP_K)
ids = tok.encode(PROMPT)

S = {}
for w in WORDS:
    for variant in (" " + w, " " + w.capitalize(), w):
        t = tok.encode(variant)
        if len(t) == 1:
            S[t[0]] = variant
print("trigger tokens:", len(S), sorted(set(S.values())))

t0 = time.time()
p, passes = exact_event_prob(next_topk, ids, H, set(S), C)
print(f"exact P(at least {C} of {H} generated tokens in trigger set) = {p:.6e}")
print(f"model passes: {passes}, time: {time.time()-t0:.1f}s")

json.dump({"prompt": PROMPT, "H": H, "C": C, "top_k": TOP_K,
           "trigger_ids": sorted(S), "exact_p": p}, open("exact_truth.json", "w"), indent=1)
print("saved exact_truth.json")


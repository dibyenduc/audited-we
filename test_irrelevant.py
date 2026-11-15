import json, sys
from real_audit_core import commit, run_log, publish, check_bookkeeping, check_cells, all_cells
from audit_strategies import relevant_cells

def main(next_topk, prompt_ids, trig, env, A=10, top_k=50, per_bin=2530):
    cfg0 = {"prompt_ids": prompt_ids, "trigger_ids": trig, "H": 3, "per_bin": per_bin,
            "n0": per_bin, "top_k": top_k, "env": env}
    same = book = rel_ok = full_flag = forged_total = 0
    for a in range(A):
        cfg = dict(cfg0, label=f"irr-{a}"); c = commit(cfg)
        h = publish(cfg, run_log(cfg, next_topk)[0])
        cps, forged = run_log(cfg, next_topk, mode="irrelevant")
        f = publish(cfg, cps); forged_total += forged
        same += f["claimed"] == h["claimed"]
        book += check_bookkeeping(f, c)[0]
        rel_ok += check_cells(f, next_topk, relevant_cells(f))
        full_flag += not check_cells(f, next_topk, all_cells(f))
    print(f"transcripts with only event-irrelevant cells forged: {A} (about {forged_total // A} forged cells each)")
    print(f"claimed estimate identical to the honest run:         {same}/{A}")
    print(f"bookkeeping passes:                                    {book}/{A}")
    print(f"feasibility-pruned audit accepts:                      {rel_ok}/{A}")
    print(f"full replay of every cell flags the forgery:           {full_flag}/{A}")

if __name__ == "__main__":
    import torch
    from lm import load_lm, MODEL
    A = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    per_bin = int(sys.argv[2]) if len(sys.argv) > 2 else 2530
    info = json.load(open("exact_truth.json"))
    tok, next_topk = load_lm(info["top_k"])
    main(next_topk, tok.encode(info["prompt"]), info["trigger_ids"],
         f"torch {torch.__version__}; {MODEL}", A, info["top_k"], per_bin)


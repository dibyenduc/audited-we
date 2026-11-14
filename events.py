def exact_event_prob(next_topk, prompt_ids, H, S, c):
    passes = [0]
    def rec(prefix, count, depth):
        remaining = H - depth
        if count >= c:
            return 1.0
        if count + remaining < c:
            return 0.0
        toks, probs = next_topk(prefix)
        passes[0] += 1
        total = 0.0
        for tk, pr in zip(toks, probs):
            hit = 1 if int(tk) in S else 0
            if remaining == 1:
                if count + hit >= c:
                    total += float(pr)
            else:
                total += float(pr) * rec(prefix + [int(tk)], count + hit, depth + 1)
        return total
    p = rec(list(prompt_ids), 0, 0)
    return p, passes[0]


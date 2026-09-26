# P2 interview study guide

## Two-minute account

I converted a post-hoc cold-start pattern into a new preregistered protocol. I trained sequence, content, and support-aware routing models on one RTX GPU, froze a three-seed ensemble, retained exact top-k artifacts, and opened the final role once. The router had a positive zero-support point estimate, but its confidence interval touched zero and warm noninferiority failed. I retained the baseline and preserved the negative result.

## Deep-dive prompts

- Why conjunctive gates? Cold improvement was not allowed to hide warm collapse.
- Why exact ranking? Sampled ranking can alter model ordering and was not the headline protocol.
- Why no reliability claim? Eleven robustness cells moved routing against the declared direction.
- Why are ablations invalid? Their executable runner was missing from the pre-lock freeze, so implementing it after outcomes would violate the protocol.

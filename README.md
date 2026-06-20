# MirrorProbe

Open research project investigating whether sycophancy under social
pressure leaves a detectable signature in LLM internal activations,
separate from output text.

This is an in-progress, honestly-reported experiment - not a finished
product. Daily build log in /logs.

## Status
- Day 1: Environment setup, GitHub repo live.
- Day 2: Loaded GPT-2 Small via TransformerLens, extracted and
  visualized a real internal activation tensor (768-dim residual
  stream, layer 6). Confirmed pipeline works end to end.
- Day 3: (in progress)

## What this is NOT yet
No sycophancy experiment has been run yet - Day 1-2 was purely
infrastructure and proving the activation-extraction mechanism works.
The actual contrastive pressure-vs-neutral experiment starts Day 3+.

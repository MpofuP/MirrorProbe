# MirrorProbe

Open research project investigating whether sycophancy under social
pressure leaves a detectable signature in LLM internal activations,
separate from output text.

This is an in-progress, honestly-reported experiment - not a finished
product. Daily build log in /logs.

## Status (Day 6)

A linear probe trained on GPT-2 Small's residual stream activations
distinguishes social-pressure framing from neutral framing with
80-97% accuracy across layers, well above a random-label control
(~50%). The result has now passed four independent checks:

- **Control 1 (lexical confound):** Decoys mentioning an authority
  figure WITHOUT pressure are classified near chance (48-54%) at
  layers 7-11, though earlier layers (0-3) show more lexical
  sensitivity (56-84%).
- **Control 2 (output-identical pairs):** Even on the subset of pairs
  where the model's generated text was nearly identical between
  conditions, the probe still distinguished internals at 87-97%
  accuracy. Caveat: GPT-2 Small is a base model and produced largely
  degenerate "I'm not sure" filler as its "identical" output, which
  weakens how strong this specific test is - a cleaner version on an
  instruction-tuned model is planned.
- **Control 3 (cross-topic generalization):** Leave-one-topic-cluster-out
  testing shows 83-96% accuracy on topics never seen during training -
  the probe is not just memorizing topic-specific vocabulary.
- **Control 4 (second model):** The same pattern replicates on
  Pythia-160M (a different architecture and training run), with real
  accuracy 70-97% vs. shuffled control 40-50% at nearly every layer.
  One anomaly (layer 1 control scored 0.80) is flagged for follow-up
  with multiple random seeds before being dismissed as noise.

## What this does NOT yet claim

This is not evidence of "genuine vs performed alignment" or any
general claim about model welfare. It is a narrower, falsifiable
finding: linear probes on residual stream activations can detect
social-pressure framing with high accuracy, robust to several
confound checks, on two small open models. Multiple random seeds,
a larger model, and an instruction-tuned model for a cleaner Control 2
are the next steps before this is written up formally.

## Project structure
- `src/` - all experiment code, by day
- `data/` - prompt pairs, extracted activations, results
- `logs/` - full build log (not pushed, local only)
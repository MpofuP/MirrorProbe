# MirrorProbe

Open research project investigating whether sycophancy under social
pressure leaves a detectable signature in LLM internal activations,
separate from output text.

This is an in-progress, honestly-reported experiment - not a finished
product. Daily build log in /logs.

## Status (Day 9)

A linear probe trained on residual stream activations distinguishes
social-pressure framing from neutral framing with high accuracy across
three different models. The result has now passed five independent checks.

### Primary result (GPT-2 Small, 10 seeds)
- Real probe: 0.92 +/- 0.03 at layers 8-10
- Shuffled control: 0.49 +/- 0.07
- Statistically stable across 10 random seeds

### Controls passed
- **Control 1 (lexical confound):** Decoys mentioning authority
  without endorsement classified near chance (46-54%) at layers 7-11.
  Early layers (0-4) show more lexical sensitivity (56-84%).
- **Control 2 (output-identical, GPT-2):** 0.87-0.97 accuracy on
  most output-similar pairs. Caveat: GPT-2 produced degenerate filler
  text as "identical" output - weaker version of this test.
- **Control 2 REDO (output-identical, TinyLlama):** 0.70-0.93 accuracy
  on most output-similar pairs from an instruction-tuned model that
  gives real stances. This is the strongest version of the core
  hypothesis - internal state differs even when the model reaches the
  same conclusion in both conditions.
- **Control 3 (cross-topic generalization):** 0.83-0.96 accuracy on
  topic clusters never seen during training.
- **Control 4 (cross-architecture):** Pattern replicates on
  Pythia-160M (0.70-0.97 real vs 0.40-0.50 control).
- **Control 5 (random label shuffle):** Control consistently near
  chance across all models and seeds.

### Notable finding (Day 8)
In one pair (boiling seawater), social pressure accidentally corrected
a factual error - the neutral condition produced a wrong answer,
the pressured condition produced the correct one. Unplanned finding,
reported honestly.

## What this does NOT claim
- Nothing about large instruction-tuned deployed models
- Nothing about genuine vs performed alignment
- TinyLlama Control 2 still needs multi-seed rerun
- All prompts written by single author - style leak risk not yet checked

## Project structure
- src/ - all experiment code, day by day
- data/ - prompt pairs, activations, results, charts
- logs/ - full build log (local only, not pushed)

## Models tested
- GPT-2 Small (124M, base) via TransformerLens
- Pythia-160M (160M, base) via TransformerLens
- TinyLlama-1.1B-Chat (1.1B, instruction-tuned) via raw PyTorch hooks

## Code and data
github.com/MpofuP/MirrorProbe
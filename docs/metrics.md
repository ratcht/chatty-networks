# Aim metrics

Every metric `ensemble/train.py` tracks to Aim, generated from `ensemble/metric_docs.py` — the same registry every run writes into `run["metric_docs"]`, so this page and the dashboard never drift apart.

## Per-run metrics

Tracked within a single `communicative` training run, grouped by what question each answers.

### Task performance

Traditional ML signal — is the ensemble learning, how accurate is it. Method-agnostic; would apply even without communication.

#### `final_test_accuracy`

**Summary.** Held-out test-set accuracy at the trained k_rounds, evaluated once after training finishes.

**Computation.** evaluate() called once on test_loader at args.k_rounds after the training loop completes.

**Intent.** The number that goes in a results table — test, not val, and evaluated exactly once so it can't be implicitly selected on. Also stored in run['final']['test_accuracy'] for cross-run aggregation.

Tracked with no context — one series.

#### `final_test_loss`

**Summary.** Held-out test-set loss at the trained k_rounds, evaluated once after training finishes.

**Computation.** Same evaluate() call as final_test_accuracy; the loss component of the same result dict.

**Intent.** Reportable end-of-run number. Also stored in run['final']['test_loss'] for cross-run aggregation.

Tracked with no context — one series.

#### `final_val_accuracy`

**Summary.** Validation accuracy at the trained k_rounds, evaluated once after training finishes.

**Computation.** evaluate() called once on val_loader at args.k_rounds after the training loop completes; tracked with a final_ prefix so it lands as a single point rather than fragmenting the per-epoch val_accuracy series.

**Intent.** The reportable end-of-run number. Also duplicated into run['final']['val_accuracy'] (not just .track()'d) specifically so summarize_replicates can read it back across a seed-replicate group with a plain dict lookup instead of a metric-sequence query.

Tracked with no context — one series.

#### `final_val_loss`

**Summary.** Validation loss at the trained k_rounds, evaluated once after training finishes.

**Computation.** Same evaluate() call as final_val_accuracy; the loss component of the same result dict.

**Intent.** Reportable end-of-run number. Also stored in run['final']['val_loss'] for the same cross-run-aggregation reason as final_val_accuracy.

Tracked with no context — one series.

#### `train_accuracy`

**Summary.** Training-set accuracy over one full epoch.

**Computation.** Running correct/seen accumulated over every training batch in the epoch, tracked once at epoch end.

**Intent.** Standard epoch-level fit signal — paired with val_accuracy to read the train/val gap.

Tracked with no context — one series.

#### `train_loss`

**Summary.** Per-step training loss.

**Computation.** NLLLoss over the prob-averaged ensemble readout — log(mean_i softmax(logits_i)) — computed once per training batch, right after the forward pass.

**Intent.** The signal actually being minimized. Track for basic health: spikes, NaNs, plateaus. Everything else in this registry exists because this number alone can't tell you *why* training is or isn't working.

Tracked with no context — one series.

#### `val_accuracy`

**Summary.** Full validation-set accuracy, evaluated once per epoch.

**Computation.** A complete forward pass over the entire val loader in eval mode, at the k_rounds the run is training with.

**Intent.** The epoch-level generalization signal — unlike the streamed single-batch val_loss, this uses the full val set, so it's the one to trust for overfitting checks or early stopping.

Tracked with no context — one series.

#### `val_loss`

**Summary.** Single-batch streamed validation loss.

**Computation.** One batch drawn per training step from an endless cycling val-loader iterator, forward pass under no_grad using the same criterion as training.

**Intent.** Gives val_loss the same sampling frequency and single-batch noise profile as train_loss — lets train/val divergence be read at training-step resolution instead of only once per epoch, at the cost of one extra forward pass per step rather than a full validation sweep.

Tracked with no context — one series.

### Communication content

The substance of what's transmitted — send (value) → combine (aggregated) → decode (injection) — independent of what any specialist does with it.

#### `msg_norm`

**Summary.** Mean L2 norm of communication-round tensors, split by pipeline stage.

**Computation.** Computed inside Orchestrator.forward after each communication round: value = V.norm(dim=-1).mean(); aggregated = c.norm(dim=-1).mean() (the post-attention output of the bus); injection = the mean, over specialists, of the decoder's output norm.

**Intent.** Localizes *where* messages vanish, if they do — value vs. aggregated vs. injection pin the collapse to sending, attention-averaging, or decoding respectively, rather than just knowing communication went quiet somewhere.

**Context — `part`.** This name covers several series, one per value:

| `part` value | meaning |
|---|---|
| `value` | what a specialist sends, before aggregation — the raw V output of its QKVEncoder. |
| `aggregated` | what a specialist receives after attention — the softmax-weighted combination of every sender's value, specific to that receiver's query. |
| `injection` | what the decoder actually adds back into the backbone, after transforming the aggregated message. A collapse toward zero at 'value' means specialists stopped saying anything meaningful; a collapse only at 'aggregated' means attention is washing messages out in the combination; a collapse only at 'injection' means the decoder itself is suppressing them before they reach the backbone. |

### Communication interpretation

How specialists' predictions/beliefs change on receiving a message.

#### `answer_shift`

**Summary.** How specialists changed their predicted class in response to one communication round, per training batch.

**Computation.** answer_shift_stats(pre, post, y) compares each specialist's pre-round and post-round argmax (captured via Orchestrator.last_shift) against the label, per batch.

**Intent.** corrected-corrupted is communication's realized, per-specialist accuracy value — the thing you actually want positive and growing. agreement_pre/post is a homogenization guard against specialists just converging on each other. See answer_shift_kl for shifts too small to flip an argmax.

**Context — `kind`.** This name covers several series, one per value:

| `kind` value | meaning |
|---|---|
| `flip_rate` | fraction of specialists whose predicted class changed at all across the round, regardless of direction. |
| `corrected` | fraction of specialists wrong before the round, right after. |
| `corrupted` | fraction of specialists right before the round, wrong after. corrected minus corrupted is the round's net, realized accuracy contribution — the number that answers whether communication is actually helping. |
| `agreement_pre` | mean pairwise agreement between specialists' predicted classes, before the round. |
| `agreement_post` | mean pairwise agreement between specialists' predicted classes, after the round. Rising agreement_post without a rising corrected-corrupted gap means specialists are converging on each other's answers, not on the truth — consensus without correctness. |

#### `answer_shift_kl`

**Summary.** Mean KL(post ‖ pre) over specialist output distributions, per training batch.

**Computation.** F.kl_div-equivalent computed directly in answer_shift_stats from the pre/post softmax distributions, averaged over specialists and the batch.

**Intent.** Catches probability shifts too small to flip an argmax, which answer_shift's flip_rate (and therefore corrected/corrupted) would miss entirely — a flip_rate of 0 doesn't mean communication had zero effect if this is nonzero.

Tracked with no context — one series.

#### `val_answer_shift`

**Summary.** Same quantities as answer_shift, computed over the full validation pass once per epoch instead of per training batch.

**Computation.** answer_shift_stats() applied to pre/post logits captured during the full-val evaluation pass in _evaluate_with_shift(), accumulated (weighted by batch size) across the entire val set.

**Intent.** The lower-variance, epoch-level counterpart to answer_shift — use this to judge communication's value at the end of an epoch rather than reading noisy per-batch swings.

**Context — `kind`.** This name covers several series, one per value:

| `kind` value | meaning |
|---|---|
| `flip_rate` | fraction of specialists whose predicted class changed at all across the round, regardless of direction. |
| `corrected` | fraction of specialists wrong before the round, right after. |
| `corrupted` | fraction of specialists right before the round, wrong after. corrected minus corrupted is the round's net, realized accuracy contribution — the number that answers whether communication is actually helping. |
| `agreement_pre` | mean pairwise agreement between specialists' predicted classes, before the round. |
| `agreement_post` | mean pairwise agreement between specialists' predicted classes, after the round. Rising agreement_post without a rising corrected-corrupted gap means specialists are converging on each other's answers, not on the truth — consensus without correctness. |

#### `val_answer_shift_kl`

**Summary.** Epoch-level counterpart to answer_shift_kl, computed over the full validation pass.

**Computation.** Same KL computation as answer_shift_kl, accumulated over the full-val evaluation pass instead of one training batch.

**Intent.** Same role as answer_shift_kl — catches sub-argmax-flip shifts — but at the lower-variance, once-per-epoch resolution.

Tracked with no context — one series.

### Training dynamics

Whether optimization itself is healthy — orthogonal to whether communication helps or the model is accurate.

#### `grad_norm`

**Summary.** L2 norm of gradients after backward, total and per component.

**Computation.** torch.norm(stack([p.grad.norm(2) for p in group])), computed once per step right after loss.backward() and before the optimizer step, grouped by component via grad_norms().

**Intent.** Earliest diagnostic for vanishing/exploding gradients, component by component — watch query_head/key_head specifically as the canary, since they're gated by the attention softmax.

**Context — `component`.** This name covers several series, one per value:

| `component` value | meaning |
|---|---|
| `total` | every trainable parameter combined. |
| `query_head` | the receiver projection in a specialist's QKVEncoder. Only receives gradient through the attention softmax, so it's the earliest warning if attention saturates and stops passing gradient back. |
| `key_head` | the signature projection in a specialist's QKVEncoder. Same softmax-only gradient path as query_head — the two are the first components to go quiet. |
| `value_head` | the sender projection in a specialist's QKVEncoder — what gets broadcast as a message. |
| `decoder` | the MLP that turns an aggregated message into an injection back into the backbone. |
| `fc` | the backbone's own classification head — the one part of the otherwise-frozen backbone left trainable. |

#### `update_ratio`

**Summary.** ‖Δθ‖ / ‖θ‖ per component after one optimizer step.

**Computation.** Parameters are snapshotted before optimizer.step(), then diffed against their post-step values; update_to_weight_ratios() divides the per-component diff norm by the pre-step parameter norm.

**Intent.** Whether weights are *actually* moving — immune to Adam's internal step-size rescaling, so it catches cases grad_norm can miss (a healthy-looking gradient whose adaptive step size is still ~0). ~1e-3 per step is healthy, ≲1e-6 is effectively frozen, ≳1e-1 is thrashing.

**Context — `component`.** This name covers several series, one per value:

| `component` value | meaning |
|---|---|
| `total` | every trainable parameter combined. |
| `query_head` | the receiver projection in a specialist's QKVEncoder. Only receives gradient through the attention softmax, so it's the earliest warning if attention saturates and stops passing gradient back. |
| `key_head` | the signature projection in a specialist's QKVEncoder. Same softmax-only gradient path as query_head — the two are the first components to go quiet. |
| `value_head` | the sender projection in a specialist's QKVEncoder — what gets broadcast as a message. |
| `decoder` | the MLP that turns an aggregated message into an injection back into the backbone. |
| `fc` | the backbone's own classification head — the one part of the otherwise-frozen backbone left trainable. |

## Replicate-group summary metrics

Tracked only on the distinguished `summary` run `ensemble/train.py summarize` (or `scripts/replicate_joint.py`) produces after a group of N_2 seed replicates all finish.

### `test_accuracy_mean`

**Summary.** Mean of final_test_accuracy across a joint-training replicate group's N_2 seeds.

**Computation.** summarize_replicates() collects final_test_accuracy (run['final']['test_accuracy']) from every run in the group's Aim experiment whose hparams.seed is in the target seed list, then takes the mean across those N_2 values.

**Intent.** A single run's final metric is one draw from joint-training seed noise. This is the number to actually report or compare across configs — the seed-averaged estimate.

Tracked with no context — one series.

### `test_accuracy_std`

**Summary.** Std of final_test_accuracy across a joint-training replicate group's N_2 seeds.

**Computation.** summarize_replicates() collects final_test_accuracy (run['final']['test_accuracy']) from every run in the group's Aim experiment whose hparams.seed is in the target seed list, then takes the population std (np.std, ddof=0) across those N_2 values.

**Intent.** A single run's final metric is one draw from joint-training seed noise. This is the number to actually report or compare across configs — how much that estimate is worth trusting: a large std relative to a difference between two configs means the difference could be noise.

Tracked with no context — one series.

### `test_loss_mean`

**Summary.** Mean of final_test_loss across a joint-training replicate group's N_2 seeds.

**Computation.** summarize_replicates() collects final_test_loss (run['final']['test_loss']) from every run in the group's Aim experiment whose hparams.seed is in the target seed list, then takes the mean across those N_2 values.

**Intent.** A single run's final metric is one draw from joint-training seed noise. This is the number to actually report or compare across configs — the seed-averaged estimate.

Tracked with no context — one series.

### `test_loss_std`

**Summary.** Std of final_test_loss across a joint-training replicate group's N_2 seeds.

**Computation.** summarize_replicates() collects final_test_loss (run['final']['test_loss']) from every run in the group's Aim experiment whose hparams.seed is in the target seed list, then takes the population std (np.std, ddof=0) across those N_2 values.

**Intent.** A single run's final metric is one draw from joint-training seed noise. This is the number to actually report or compare across configs — how much that estimate is worth trusting: a large std relative to a difference between two configs means the difference could be noise.

Tracked with no context — one series.

### `val_accuracy_mean`

**Summary.** Mean of final_val_accuracy across a joint-training replicate group's N_2 seeds.

**Computation.** summarize_replicates() collects final_val_accuracy (run['final']['val_accuracy']) from every run in the group's Aim experiment whose hparams.seed is in the target seed list, then takes the mean across those N_2 values.

**Intent.** A single run's final metric is one draw from joint-training seed noise. This is the number to actually report or compare across configs — the seed-averaged estimate.

Tracked with no context — one series.

### `val_accuracy_std`

**Summary.** Std of final_val_accuracy across a joint-training replicate group's N_2 seeds.

**Computation.** summarize_replicates() collects final_val_accuracy (run['final']['val_accuracy']) from every run in the group's Aim experiment whose hparams.seed is in the target seed list, then takes the population std (np.std, ddof=0) across those N_2 values.

**Intent.** A single run's final metric is one draw from joint-training seed noise. This is the number to actually report or compare across configs — how much that estimate is worth trusting: a large std relative to a difference between two configs means the difference could be noise.

Tracked with no context — one series.

### `val_loss_mean`

**Summary.** Mean of final_val_loss across a joint-training replicate group's N_2 seeds.

**Computation.** summarize_replicates() collects final_val_loss (run['final']['val_loss']) from every run in the group's Aim experiment whose hparams.seed is in the target seed list, then takes the mean across those N_2 values.

**Intent.** A single run's final metric is one draw from joint-training seed noise. This is the number to actually report or compare across configs — the seed-averaged estimate.

Tracked with no context — one series.

### `val_loss_std`

**Summary.** Std of final_val_loss across a joint-training replicate group's N_2 seeds.

**Computation.** summarize_replicates() collects final_val_loss (run['final']['val_loss']) from every run in the group's Aim experiment whose hparams.seed is in the target seed list, then takes the population std (np.std, ddof=0) across those N_2 values.

**Intent.** A single run's final metric is one draw from joint-training seed noise. This is the number to actually report or compare across configs — how much that estimate is worth trusting: a large std relative to a difference between two configs means the difference could be noise.

Tracked with no context — one series.

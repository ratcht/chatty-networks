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

#### `attn_entropy`

**Summary.** Mean per-row (per-receiver) entropy of the bus's attention weights over senders, averaged across the batch.

**Computation.** -(attn * attn.log()).sum(-1) computed on TarMACBus's post-softmax attn (b, n, n) inside Orchestrator.forward, then meaned over both the receiver and batch dims to one scalar per step.

**Intent.** How concentrated each receiver's attention is over the n senders. Low and falling means receivers are converging onto ~one sender each (a de-facto hard selection); high and flat near log(n) (uniform attention) means every receiver is just averaging all senders equally, which — combined with a flat msg_norm — would suggest attention isn't discriminating at all.

Tracked with no context — one series.

#### `attn_top1`

**Summary.** Mean peak attention weight, max_i p_i, over the bus's post-softmax attention rows, averaged across receivers and the batch.

**Computation.** attn.max(dim=-1).values.mean() on TarMACBus's (b, n, n) post-softmax attn inside Orchestrator.forward, meaned over the receiver and batch dims to one scalar per step — the same rows attn_entropy summarizes.

**Intent.** Concentration on a scale that reads directly: 1/n is uniform, 1.0 is a hard one-hot gate, and the number is comparable across groups of different size in a way entropy's log(n) ceiling is not. Read with attn_entropy — the two separate a row with one dominant sender and a flat tail (high top1, mid entropy) from one split evenly between two senders (mid top1, mid entropy), which entropy alone conflates.

Tracked with no context — one series.

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

#### `routing_mode_share`

**Summary.** Fraction of examples in the batch whose routing pattern is the single most common one.

**Computation.** routing_pattern_counts(): attn.argmax(-1) reduces each example to a tuple of attended-to senders, one per receiver (e.g. (0, 2, 2)); the tuple is encoded base-n into one integer and tallied, and the mode's count is divided by the number of examples. Per-receiver series tally that receiver's argmax alone. Histograms are pooled across batches and the mode taken once at the end, never averaged per batch — the modal pattern of a whole set need not be the modal pattern of any single batch.

**Intent.** Whether routing depends on the input at all. attn_entropy and attn_top1 both average over the batch, so a receiver that always attends 100% to expert 2 and one that attends 100% to a different, image-appropriate expert each time are indistinguishable to them — identical entropy (0.0) and top1 (1.0) for opposite outcomes. This separates them: 1.0 means one wiring for every image (the channel is a constant skip connection, and every question about *who* listens to whom is moot), while a low value means the pattern genuinely varies. Read alongside attn_top1 to rule out the false positive: a low mode share with a low top1 is argmax flipping on near-ties, i.e. noise, not routing.

**Context — `receiver`.** This name covers several series, one per value:

| `receiver` value | meaning |
|---|---|
| `joint` | the whole wiring diagram: an example counts toward the mode only if *every* receiver matches the modal tuple. The strict series, and the headline number — 1.0 means one fixed wiring for the entire dataset. Being a joint statistic it is also the harshest: it is bounded above by every per-receiver series, so a low joint value alone does not say which receiver is varying. |
| `<i>` | one series per receiver index i (0 … k-1): the mode share of that receiver's argmax on its own, ignoring what the others did. Read these against 'joint' to localize the variation — three per-receiver values near 1.0 with a low joint value would mean the receivers each have a strong preference but disagree about which example is the exception. |

#### `self_attn_rate`

**Summary.** Fraction of (example, receiver) pairs whose argmax attention is the receiver's own message.

**Computation.** routing_counts() takes attn.argmax(-1) over the bus's (b, n, n) attention and counts how often a receiver's peak lands on its own index, over all receivers and examples in the batch. Tracked per step on train batches and per epoch over the full val set (val_self_attn_rate).

**Intent.** TarMACBus lets every member attend over all members including itself, so a receiver can route to its own message. High self-attention means communication is largely not happening — the receiver is reading itself back. It is also the confound that makes routed_correct's corrected split circular, which is why routed_correct excludes these pairs.

Tracked with no context — one series.

#### `val_route_flip_rate`

**Summary.** Fraction of (example, receiver) pairs whose attended-to sender changed since the previous epoch.

**Computation.** _evaluate_with_shift() records attn.argmax(-1) for the whole val set as an (N, k) tensor in loader order; route_flip_rate() diffs it against the previous epoch's and takes the mean. The comparison is well-defined because the val loader is shuffle=False over a fixed Subset with no augmentation, so row r is the same image every epoch. Not tracked at epoch 0 — there is nothing to diff against, and a 0.0 there would read as 'routing never moved', the exact opposite of the truth.

**Intent.** Whether the wiring was learned or decided at initialization. routing_mode_share is a snapshot: a reading of 1.0 at the last epoch is equally consistent with routing that explored early and then collapsed (an architecture/objective problem — fix with temperature or entropy regularization) and with routing that was set by random init and never moved because the softmax saturated and the query/key gradients died (a training-dynamics problem — fix with key/query normalization or score scaling). Same endpoint, opposite remedies, and only the trajectory distinguishes them: high-then-decaying is the first, flat at ~0 from epoch 1 is the second. Note the Decoder's output layer is zero-initialized, so the injected message starts at exactly zero and attention gets no learning signal for the first steps — the second story is live. Reference points for k=3: 0.0 is identical to last epoch, 0.667 is as different as re-randomizing. Cross-check a high value against attn_top1 — flipping among near-ties is tie noise, not re-routing.

Tracked with no context — one series.

#### `val_routing_mode_share`

**Summary.** Same quantity as routing_mode_share, over the full validation pass once per epoch instead of per training batch.

**Computation.** routing_pattern_counts() applied during _evaluate_with_shift(), with the pattern histograms summed across every batch and the mode taken over the whole val set.

**Intent.** The series to read for anything quantitative. Mode share is biased upward on small samples — at batch size 1 it is trivially 1.0, and even at 128 a genuinely random routing floors out near 0.09 rather than the 0.04 it approaches over the full val set. So the per-step train series is a within-run trend line only; its absolute level is not comparable to this one, and would shift if the batch size changed.

**Context — `receiver`.** This name covers several series, one per value:

| `receiver` value | meaning |
|---|---|
| `joint` | the whole wiring diagram: an example counts toward the mode only if *every* receiver matches the modal tuple. The strict series, and the headline number — 1.0 means one fixed wiring for the entire dataset. Being a joint statistic it is also the harshest: it is bounded above by every per-receiver series, so a low joint value alone does not say which receiver is varying. |
| `<i>` | one series per receiver index i (0 … k-1): the mode share of that receiver's argmax on its own, ignoring what the others did. Read these against 'joint' to localize the variation — three per-receiver values near 1.0 with a low joint value would mean the receivers each have a strong preference but disagree about which example is the exception. |

#### `val_self_attn_rate`

**Summary.** Same quantity as self_attn_rate, over the full validation pass once per epoch instead of per training batch.

**Computation.** routing_counts() applied to the attention and pre/post logits captured during _evaluate_with_shift(), with numerators and denominators pooled across the whole val set.

**Intent.** The lower-variance, epoch-level counterpart to self_attn_rate — the one to read when judging whether receivers are talking to each other or to themselves.

Tracked with no context — one series.

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

#### `routed_correct`

**Summary.** How often the specialist a receiver attended to was itself right about the example — over pairs that attended to someone else.

**Computation.** routing_counts(): attn.argmax(-1) picks each receiver's attended-to sender, which is looked up against that sender's own pre-communication correctness (Orchestrator.last_shift's 'pre' argmax vs the label). Self-attending pairs are dropped. Aggregated as pooled counts, not as a mean of per-batch rates, so uneven subset sizes weight correctly.

**Intent.** Separates routing sharply from routing *well* — attn_entropy cannot tell a confident-and-right gate from a confident-and-wrong one. Read against the mean specialist accuracy: at that level, attention carries no information about who is right. The corrected/corrupted split is the sharper test, asking whether routing quality tracks the good and harm communication actually did on that example rather than just correlating in aggregate.

**Context — `subset`.** This name covers several series, one per value:

| `subset` value | meaning |
|---|---|
| `all` | every (example, receiver) pair where the receiver attended to another specialist. The headline routing-quality number: compare it against the mean specialist accuracy, which is what routing at random would score. |
| `corrected` | the subset of those pairs where communication flipped this receiver from wrong to right. If routing is doing the work, this should sit above the 'all' rate — the receiver listened to someone who knew the answer. |
| `corrupted` | the subset where communication flipped this receiver from right to wrong. Should sit *below* the 'all' rate: the damage should be traceable to having listened to a specialist that was itself wrong. If corrected and corrupted are both at the 'all' rate, routing quality is unrelated to what communication actually did. |

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

#### `val_routed_correct`

**Summary.** Same quantity as routed_correct, over the full validation pass once per epoch instead of per training batch.

**Computation.** routing_counts() applied over _evaluate_with_shift()'s pass, summing counts rather than averaging per-batch rates — the corrected/corrupted subsets are small and uneven per batch, so only a pooled numerator/denominator gives the right weighting.

**Intent.** The number to actually judge routing quality on: per-batch corrected/corrupted subsets are far too small to read directly. Compare against the mean specialist accuracy for this group.

**Context — `subset`.** This name covers several series, one per value:

| `subset` value | meaning |
|---|---|
| `all` | every (example, receiver) pair where the receiver attended to another specialist. The headline routing-quality number: compare it against the mean specialist accuracy, which is what routing at random would score. |
| `corrected` | the subset of those pairs where communication flipped this receiver from wrong to right. If routing is doing the work, this should sit above the 'all' rate — the receiver listened to someone who knew the answer. |
| `corrupted` | the subset where communication flipped this receiver from right to wrong. Should sit *below* the 'all' rate: the damage should be traceable to having listened to a specialist that was itself wrong. If corrected and corrupted are both at the 'all' rate, routing quality is unrelated to what communication actually did. |

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
| `trunk` | the shared MLP trunk before it diverges into query/key/value heads. Present for --encoder mlp (Linear->GELU->LayerNorm) and --encoder mlp_no_ln (Linear->GELU, no LayerNorm) — not for qkv, which has no shared trunk at all. |

#### `layernorm_param_norm`

**Summary.** L2 norm of every LayerNorm's affine parameters, gamma and beta.

**Computation.** torch.norm(stack([p.norm(2) for p in group])) over every nn.LayerNorm.weight (gamma) / .bias (beta) in the model, computed once per step via layernorm_param_norms(). Only non-empty for --encoder mlp — MLPEncoder's trunk is the only LayerNorm in the model.

**Intent.** Whether the trunk's LayerNorm is actually learning a non-trivial affine transform or sitting at its identity init (gamma≈1 per element, beta≈0) — a LayerNorm stuck at init is doing pure normalization with no learned rescaling, which would undercut the expressiveness MLPEncoder is supposed to add over QKVEncoder.

**Context — `param`.** This name covers several series, one per value:

| `param` value | meaning |
|---|---|
| `gamma` | LayerNorm's learned scale (nn.LayerNorm.weight), one per normalized feature. Identity init is all-ones, so a norm near sqrt(hidden_dim) means it hasn't moved from init; drifting away means the trunk is learning a non-trivial rescaling. |
| `beta` | LayerNorm's learned shift (nn.LayerNorm.bias), one per normalized feature. Identity init is all-zeros, so a norm near zero means it hasn't moved from init. |

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
| `trunk` | the shared MLP trunk before it diverges into query/key/value heads. Present for --encoder mlp (Linear->GELU->LayerNorm) and --encoder mlp_no_ln (Linear->GELU, no LayerNorm) — not for qkv, which has no shared trunk at all. |

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

# Communication audit plan

Are the specialists in the communicative ensemble actually exchanging useful
information, or is the reported gain just the trainable `fc` heads fine-tuning?

## Motivating evidence

Metadata of the one trained checkpoint
(`ensemble/checkpoints/communicative_snapshot_backbone_seed42_snap3of5_snapshot_backbone_seed42_snap5of5.pt`,
70 epochs, `k_rounds=2`):

| | val acc | test acc | val loss |
|---|---|---|---|
| snap3 alone | 0.5882 | — | — |
| snap5 alone | 0.6028 | — | — |
| plain prob-average (untrained, `k_rounds=0`) | 0.6100 | 0.5984 | 1.850 |
| trained communicative (`k_rounds=2`) | 0.6000 | 0.6004 | 1.564 |

Accuracy flat, loss down 1.85 → 1.56: consistent with the `fc` heads learning
calibration while the channel contributes nothing. This audit localises where the
channel fails — or shows it works and the members simply have nothing to say
(they are two snapshots of the same run, seeing the same image).

Framework (Lowe et al. 2019, arXiv:1903.05168): **positive signalling** (messages
carry information) is measured in part 1; **positive listening** (receivers act on
it) in part 2. Signalling can be high while listening is zero, so part 1 alone can
kill the mechanism but never validate it.

## What a message is

```
late features (layer3.0, 64ch 8x8) -> mean pool -> 64
  -> value_head                    -> V   (64)   sent
  -> attention mix over members    -> c   (64)   heard
  -> decoder MLP                   -> msg (16)   injected as per-channel bias
     at layer1.2 input (16ch 32x32)
```

With `k_rounds=2` there are 3 backbone passes and 2 messages. Message 1 is
produced from clean perception (pass 0) and injected into pass 1. All probing
targets message 1; message 2 is contaminated by message 1.

---

# Part 1 — how good are the messages?

Linear classifier probes (Alain & Bengio 2016, arXiv:1610.01644): a linear
classifier trained on detached features from one site; its val error measures the
linear class information present there. Softmax cross-entropy on a linear model is
convex, so a converged solver gives *the* optimal probe — the number is a property
of the features, not of probe training hyperparameters. Probes are sklearn
`make_pipeline(StandardScaler(), LogisticRegression())`; fit on
`ensemble_indices` (20k, no augmentation), report error on `val_indices` (5k) and
the CIFAR-100 test set (10k).

## 1a. Finish probing TarMAC — message quality

Sites captured from one `no_grad()` forward of the trained checkpoint:

| site | dim | meaning |
|---|---|---|
| `f` | 64 | pooled late features — what the sender knows |
| `V` | 64 | what it says |
| `c` | 64 | what the receiver hears |
| `msg` | 16 | what gets injected |

Probes:

1. `f -> y` — receiver's own ceiling
2. `V -> y` — class info in what is sent
3. `c -> y` — in what is heard
4. `msg -> y` — in what is injected
5. `[f || c] -> y` vs probe 1 — does the message add anything the receiver
   didn't already know (novelty)
6. `c -> y` on the subset where the receiver's pass-0 prediction was wrong —
   can the message fix a mistake, the core question

Controls:

- chance via `DummyClassifier(strategy="prior")`
- probes 2–3 refit on a random-init encoder — random projections probe well
  above chance, so trained-vs-random is the signal, not the absolute number
- no random control for `msg` (decoder last layer is zero-init, its random
  output is identically zero)

Reading the chain: probe error is non-decreasing along `f -> V` and `c -> msg`
(information cannot be created); the link where error collapses to chance is the
broken component. `c` beating a single `V` is fine — aggregation. If everything
probes well but probe 5's delta is ~0, the mechanism works and the members are
just redundant: a diversity problem, not a mechanism problem.

Changes: `capture` arg on `Orchestrator.forward`, `augment` flag on
`make_loaders`, new `ensemble/probe.py` runner, `scikit-learn` + `pandas` in
requirements. Runs on the CUDA box; local venv lacks torchvision/sklearn/data.

## 1b. Probe the backbones — where should the ports go?

Current ports (`early=layer1.2` write, `late=layer3.0` read) were chosen by
hand. Classic depth probing gives an empirical basis: probe every block output
of a frozen ResNet20 (`conv1`, `layer1.0–2`, `layer2.0–2`, `layer3.0–2`, pooled)
for `y`, via `torchvision`'s `create_feature_extractor`. No orchestrator
involved.

- **read port**: wants high class information — probe error curve directly
  ranks candidates. Expect monotone improvement with depth (their §4.1), so the
  question is how much is lost reading at `layer3.0` vs `layer3.2`.
- **write port**: probes inform but cannot decide — injecting where class info
  is low leaves depth to process the message, but whether the frozen stack
  *uses* it is a listening question. Pick candidates here, verify in part 2.

Also probe the *pooled* version of each site: the encoder mean-pools before
reading, and pooling may be what destroys the information rather than the depth
choice. `f` at 64-d pooled vs 4096-d unpooled tells us the cost of the current
reduce.

## 1c. Only if 1a/1b are ambiguous — probes during training

Retrain with `nn.Linear` probes attached (detached inputs, separate optimizer,
tracked in Aim like the existing diagnostics). The decoder is zero-init so its
probe starts at chance by construction; a curve that never leaves chance over 70
epochs is the dead-channel signature (their §5.1). This is the strongest
evidence but costs a training run, so it waits for the static read.

---

# Part 2 — how well are the messages used?

Causal interventions at the aggregated `c` (the four-condition design from the
latent-channel audit literature). Accuracy under each condition, same loader,
no training:

| condition | implementation | isolates |
|---|---|---|
| none | `c = 0` | communication off |
| other-example | `c = c.roll(1, dims=0)` | is the content example-specific |
| self-only | attention restricted to self | second forward pass vs a real partner |
| real | unchanged | full mechanism |

Derived numbers:

- **CIC** real vs other-example — messages causally used at all
- **CAG** real vs none — value of example-specific content
- **SSG** real vs self-only — value of the *other* member beyond a second pass

Plus:

- attention diagnostics: off-diagonal mass, and correlation of `attn_ij` with
  member `j` being correct (does anyone listen to whoever is right). Weak with
  n=2 and self-attention included; meaningful at 4–5 members.
- the honest baseline: same-epochs `k_rounds=0` run with `fc` trainable, so
  "communication helps" is no longer confounded with "fine-tuning helps". The
  current printed baseline is the *untrained* average.

Expected readings:

| signalling (p1) | listening (p2) | conclusion |
|---|---|---|
| low | — | encoder/decoder broken; fix mechanism, part 2 moot |
| high | low | content exists, receiver can't use it — injection site/strength, or frozen stack can't read the subspace |
| high | high | real communication; measure size via CAG, then attack diversity |
| high, novelty ~0 | — | mechanism fine, members redundant — need diverse backbones or different views per member |

## Known confounds

1. Members are snapshots of one run — correlated, little to disagree about.
2. All members see the same image — no private information exists; messages can
   only convey interpretation. If the mechanism is sound but useless, give
   members different views (crops/occlusions); `forward` already takes a list.
3. Probes prove presence, not usability: linear separability in a subspace the
   frozen downstream layers ignore is invisible to them. Hence part 2.

## Order

1. 1a probe chain + 1b backbone depth probes (one script run, same cache pass)
2. read out; if ambiguous, 1c training-time probes
3. part 2 interventions on the existing checkpoint
4. decide: fix mechanism, fix ports, or fix diversity
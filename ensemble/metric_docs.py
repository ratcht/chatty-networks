"""
Documentation for every metric tracked to Aim by ensemble/train.py.

Aim's dashboard shows a metric by its bare name plus whatever `context` dict
distinguishes its series (e.g. grad_norm split by component) — neither one
carries an explanation of what the numbers mean, how they're computed, or why
they're tracked. This registry is the single source of truth for that:
written into every run as run["metric_docs"] at creation, so it's readable
from the dashboard without opening source, and rendered to docs/metrics.md by
scripts/render_metric_docs.py for the MkDocs site.

Each entry is split into three questions, deliberately kept separate rather
than folded into one paragraph:
  summary     — what the number IS, in one sentence
  computation — the actual mechanism: where in the code it's computed and how
  intent      — why it's tracked: what to watch for, what good/bad looks like

A metric with a `context` (e.g. grad_norm split by component) is really
several series sharing one name, so each ContextKey documents what its
*values* mean — "component" alone tells you nothing; "component=query_head
only receives gradient through the attention softmax" does.

Mirrors the Metric-registry pattern in selection/metrics.py, but purely
descriptive — nothing here computes anything, so there's no fn/requires.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextKey:
  name: str
  values: dict[str, str]  # value -> what that series is


@dataclass(frozen=True)
class MetricDoc:
  name: str
  summary: str
  computation: str
  intent: str
  category: str
  context: tuple[ContextKey, ...] = ()


# Conceptual blocks per-run metrics fall into — what QUESTION each answers,
# not just a label. Order here is display order in docs/metrics.md.
CATEGORIES: tuple[tuple[str, str, str], ...] = (
  ("task_performance", "Task performance",
   "Traditional ML signal — is the ensemble learning, how accurate is it. "
   "Method-agnostic; would apply even without communication."),
  ("communication_content", "Communication content",
   "The substance of what's transmitted — send (value) → combine "
   "(aggregated) → decode (injection) — independent of what any specialist "
   "does with it."),
  ("communication_interpretation", "Communication interpretation",
   "How specialists' predictions/beliefs change on receiving a message."),
  ("training_dynamics", "Training dynamics",
   "Whether optimization itself is healthy — orthogonal to whether "
   "communication helps or the model is accurate."),
)


# One ContextKey shared by grad_norm and update_ratio — both split the same
# way, by which trainable component the number belongs to.
_COMPONENT = ContextKey(
  name="component",
  values={
    "total": "every trainable parameter combined.",
    "query_head": "the receiver projection in a specialist's QKVEncoder. Only "
                  "receives gradient through the attention softmax, so it's "
                  "the earliest warning if attention saturates and stops "
                  "passing gradient back.",
    "key_head": "the signature projection in a specialist's QKVEncoder. Same "
                "softmax-only gradient path as query_head — the two are the "
                "first components to go quiet.",
    "value_head": "the sender projection in a specialist's QKVEncoder — what "
                  "gets broadcast as a message.",
    "decoder": "the MLP that turns an aggregated message into an injection "
               "back into the backbone.",
    "fc": "the backbone's own classification head — the one part of the "
          "otherwise-frozen backbone left trainable.",
  },
)

_ANSWER_SHIFT_KIND = ContextKey(
  name="kind",
  values={
    "flip_rate": "fraction of specialists whose predicted class changed at "
                 "all across the round, regardless of direction.",
    "corrected": "fraction of specialists wrong before the round, right "
                 "after.",
    "corrupted": "fraction of specialists right before the round, wrong "
                 "after. corrected minus corrupted is the round's net, "
                 "realized accuracy contribution — the number that answers "
                 "whether communication is actually helping.",
    "agreement_pre": "mean pairwise agreement between specialists' predicted "
                     "classes, before the round.",
    "agreement_post": "mean pairwise agreement between specialists' predicted "
                      "classes, after the round. Rising agreement_post "
                      "without a rising corrected-corrupted gap means "
                      "specialists are converging on each other's answers, "
                      "not on the truth — consensus without correctness.",
  },
)

_MSG_NORM_PART = ContextKey(
  name="part",
  values={
    "value": "what a specialist sends, before aggregation — the raw V "
             "output of its QKVEncoder.",
    "aggregated": "what a specialist receives after attention — the "
                  "softmax-weighted combination of every sender's value, "
                  "specific to that receiver's query.",
    "injection": "what the decoder actually adds back into the backbone, "
                 "after transforming the aggregated message. A collapse "
                 "toward zero at 'value' means specialists stopped saying "
                 "anything meaningful; a collapse only at 'aggregated' means "
                 "attention is washing messages out in the combination; a "
                 "collapse only at 'injection' means the decoder itself is "
                 "suppressing them before they reach the backbone.",
  },
)


_DOCS: tuple[MetricDoc, ...] = (
  MetricDoc(
    name="train_loss",
    summary="Per-step training loss.",
    computation="NLLLoss over the prob-averaged ensemble readout — "
                "log(mean_i softmax(logits_i)) — computed once per training "
                "batch, right after the forward pass.",
    intent="The signal actually being minimized. Track for basic health: "
           "spikes, NaNs, plateaus. Everything else in this registry exists "
           "because this number alone can't tell you *why* training is or "
           "isn't working.",
    category="task_performance",
  ),
  MetricDoc(
    name="grad_norm",
    summary="L2 norm of gradients after backward, total and per component.",
    computation="torch.norm(stack([p.grad.norm(2) for p in group])), computed "
                "once per step right after loss.backward() and before the "
                "optimizer step, grouped by component via grad_norms().",
    intent="Earliest diagnostic for vanishing/exploding gradients, "
           "component by component — watch query_head/key_head specifically "
           "as the canary, since they're gated by the attention softmax.",
    category="training_dynamics",
    context=(_COMPONENT,),
  ),
  MetricDoc(
    name="update_ratio",
    summary="‖Δθ‖ / ‖θ‖ per component after one optimizer step.",
    computation="Parameters are snapshotted before optimizer.step(), then "
                "diffed against their post-step values; update_to_weight_ratios() "
                "divides the per-component diff norm by the pre-step "
                "parameter norm.",
    intent="Whether weights are *actually* moving — immune to Adam's "
           "internal step-size rescaling, so it catches cases grad_norm can "
           "miss (a healthy-looking gradient whose adaptive step size is "
           "still ~0). ~1e-3 per step is healthy, ≲1e-6 is effectively "
           "frozen, ≳1e-1 is thrashing.",
    category="training_dynamics",
    context=(_COMPONENT,),
  ),
  MetricDoc(
    name="msg_norm",
    summary="Mean L2 norm of communication-round tensors, split by pipeline "
            "stage.",
    computation="Computed inside Orchestrator.forward after each "
                "communication round: value = V.norm(dim=-1).mean(); "
                "aggregated = c.norm(dim=-1).mean() (the post-attention "
                "output of the bus); injection = the mean, over specialists, "
                "of the decoder's output norm.",
    intent="Localizes *where* messages vanish, if they do — value vs. "
           "aggregated vs. injection pin the collapse to sending, "
           "attention-averaging, or decoding respectively, rather than just "
           "knowing communication went quiet somewhere.",
    category="communication_content",
    context=(_MSG_NORM_PART,),
  ),
  MetricDoc(
    name="answer_shift",
    summary="How specialists changed their predicted class in response to "
            "one communication round, per training batch.",
    computation="answer_shift_stats(pre, post, y) compares each specialist's "
                "pre-round and post-round argmax (captured via "
                "Orchestrator.last_shift) against the label, per batch.",
    intent="corrected-corrupted is communication's realized, per-specialist "
           "accuracy value — the thing you actually want positive and "
           "growing. agreement_pre/post is a homogenization guard against "
           "specialists just converging on each other. See "
           "answer_shift_kl for shifts too small to flip an argmax.",
    category="communication_interpretation",
    context=(_ANSWER_SHIFT_KIND,),
  ),
  MetricDoc(
    name="answer_shift_kl",
    summary="Mean KL(post ‖ pre) over specialist output distributions, per "
            "training batch.",
    computation="F.kl_div-equivalent computed directly in answer_shift_stats "
                "from the pre/post softmax distributions, averaged over "
                "specialists and the batch.",
    intent="Catches probability shifts too small to flip an argmax, which "
           "answer_shift's flip_rate (and therefore corrected/corrupted) "
           "would miss entirely — a flip_rate of 0 doesn't mean "
           "communication had zero effect if this is nonzero.",
    category="communication_interpretation",
  ),
  MetricDoc(
    name="val_loss",
    summary="Single-batch streamed validation loss.",
    computation="One batch drawn per training step from an endless cycling "
                "val-loader iterator, forward pass under no_grad using the "
                "same criterion as training.",
    intent="Gives val_loss the same sampling frequency and single-batch "
           "noise profile as train_loss — lets train/val divergence be read "
           "at training-step resolution instead of only once per epoch, at "
           "the cost of one extra forward pass per step rather than a full "
           "validation sweep.",
    category="task_performance",
  ),
  MetricDoc(
    name="train_accuracy",
    summary="Training-set accuracy over one full epoch.",
    computation="Running correct/seen accumulated over every training batch "
                "in the epoch, tracked once at epoch end.",
    intent="Standard epoch-level fit signal — paired with val_accuracy to "
           "read the train/val gap.",
    category="task_performance",
  ),
  MetricDoc(
    name="val_accuracy",
    summary="Full validation-set accuracy, evaluated once per epoch.",
    computation="A complete forward pass over the entire val loader in eval "
                "mode, at the k_rounds the run is training with.",
    intent="The epoch-level generalization signal — unlike the streamed "
           "single-batch val_loss, this uses the full val set, so it's the "
           "one to trust for overfitting checks or early stopping.",
    category="task_performance",
  ),
  MetricDoc(
    name="val_answer_shift",
    summary="Same quantities as answer_shift, computed over the full "
            "validation pass once per epoch instead of per training batch.",
    computation="answer_shift_stats() applied to pre/post logits captured "
                "during the full-val evaluation pass in "
                "_evaluate_with_shift(), accumulated (weighted by batch "
                "size) across the entire val set.",
    intent="The lower-variance, epoch-level counterpart to answer_shift — "
           "use this to judge communication's value at the end of an epoch "
           "rather than reading noisy per-batch swings.",
    category="communication_interpretation",
    context=(_ANSWER_SHIFT_KIND,),
  ),
  MetricDoc(
    name="val_answer_shift_kl",
    summary="Epoch-level counterpart to answer_shift_kl, computed over the "
            "full validation pass.",
    computation="Same KL computation as answer_shift_kl, accumulated over "
                "the full-val evaluation pass instead of one training batch.",
    intent="Same role as answer_shift_kl — catches sub-argmax-flip shifts — "
           "but at the lower-variance, once-per-epoch resolution.",
    category="communication_interpretation",
  ),
  MetricDoc(
    name="final_val_accuracy",
    summary="Validation accuracy at the trained k_rounds, evaluated once "
            "after training finishes.",
    computation="evaluate() called once on val_loader at args.k_rounds after "
                "the training loop completes; tracked with a final_ prefix "
                "so it lands as a single point rather than fragmenting the "
                "per-epoch val_accuracy series.",
    intent="The reportable end-of-run number. Also duplicated into "
           "run['final']['val_accuracy'] (not just .track()'d) specifically "
           "so summarize_replicates can read it back across a seed-replicate "
           "group with a plain dict lookup instead of a metric-sequence "
           "query.",
    category="task_performance",
  ),
  MetricDoc(
    name="final_val_loss",
    summary="Validation loss at the trained k_rounds, evaluated once after "
            "training finishes.",
    computation="Same evaluate() call as final_val_accuracy; the loss "
                "component of the same result dict.",
    intent="Reportable end-of-run number. Also stored in "
           "run['final']['val_loss'] for the same cross-run-aggregation "
           "reason as final_val_accuracy.",
    category="task_performance",
  ),
  MetricDoc(
    name="final_test_accuracy",
    summary="Held-out test-set accuracy at the trained k_rounds, evaluated "
            "once after training finishes.",
    computation="evaluate() called once on test_loader at args.k_rounds "
                "after the training loop completes.",
    intent="The number that goes in a results table — test, not val, and "
           "evaluated exactly once so it can't be implicitly selected on. "
           "Also stored in run['final']['test_accuracy'] for cross-run "
           "aggregation.",
    category="task_performance",
  ),
  MetricDoc(
    name="final_test_loss",
    summary="Held-out test-set loss at the trained k_rounds, evaluated once "
            "after training finishes.",
    computation="Same evaluate() call as final_test_accuracy; the loss "
                "component of the same result dict.",
    intent="Reportable end-of-run number. Also stored in "
           "run['final']['test_loss'] for cross-run aggregation.",
    category="task_performance",
  ),
)

REGISTRY: dict[str, MetricDoc] = {d.name: d for d in _DOCS}


def _serialize(d: MetricDoc) -> dict:
  return {
    "summary": d.summary,
    "computation": d.computation,
    "intent": d.intent,
    "category": d.category,
    "context": [{"name": c.name, "values": c.values} for c in d.context],
  }


def as_dict() -> dict[str, dict]:
  """Serialize the registry for run["metric_docs"] — plain str/list/dict,
  exactly what Aim's param storage can hold."""
  return {d.name: _serialize(d) for d in _DOCS}


# ---------------------------------------------------------------------------
# Stage-4 summary metrics — a statistic over a replicate group's runs, not a
# value produced within any single one. Extends this same registry rather
# than living in a separate one; scripts/replicate_joint.py trains the runs,
# ensemble/train.py:summarize_replicates aggregates them.
# ---------------------------------------------------------------------------

_SUMMARY_DOCS: tuple[MetricDoc, ...] = tuple(
  MetricDoc(
    name=f"{base}_{stat}",
    summary=f"{stat.capitalize()} of final_{base} across a joint-training "
            f"replicate group's N_2 seeds.",
    computation=f"summarize_replicates() collects final_{base} "
                f"(run['final']['{base}']) from every run in the group's "
                f"Aim experiment whose hparams.seed is in the target seed "
                f"list, then takes the {'mean' if stat == 'mean' else 'population std (np.std, ddof=0)'} "
                f"across those N_2 values.",
    intent="A single run's final metric is one draw from joint-training "
           f"seed noise. This is the number to actually report or compare "
           f"across configs — {'the seed-averaged estimate' if stat == 'mean' else 'how much that estimate is worth trusting: a large std relative to a difference between two configs means the difference could be noise'}.",
    category="replicate_summary",
  )
  for base in ("val_accuracy", "val_loss", "test_accuracy", "test_loss")
  for stat in ("mean", "std")
)

SUMMARY_REGISTRY: dict[str, MetricDoc] = {d.name: d for d in _SUMMARY_DOCS}


def summary_as_dict() -> dict[str, dict]:
  return {d.name: _serialize(d) for d in _SUMMARY_DOCS}

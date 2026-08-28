"""
Train an ensemble over frozen backbones, given a split and backbone checkpoints.

Methods (one subcommand each; gating will join later):

communicative — TarMAC-aligned ensemble: per-specialist Q/K/V heads, bare
                attention bus, decoder injection via hooks.

A `summarize` subcommand aggregates a group of same-config, different-seed
replicate runs (produced by scripts/replicate_joint.py) into one summary run.

Example
-------
uv run python ensemble/train.py communicative \\
    --manifest selection/manifests/high_gap.json \\
    --split-file splits/three_way_seed42.pt \\
    --epochs 10 --k-rounds 1
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from aim import Repo, Run
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backbone.resnet import ResNet
from ensemble.communicative.driver import Driver
from ensemble.communicative.bus import QKVEncoder, MLPEncoder, Decoder
from ensemble.communicative.orchestrator import Specialist, Orchestrator
from ensemble.eval import evaluate
from ensemble import metric_docs

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def seed_everything(seed: int, deterministic: bool) -> None:
  """Seed every RNG a training run can touch, including ones nothing uses yet.

  cudnn is the one that matters most here: left alone it's free to pick a
  different conv algorithm run-to-run even at a fixed seed, which on a
  54-block ResNet-110 stack is a second, unseeded source of variance sitting
  on top of the joint-training seed this whole replicate study exists to
  isolate. deterministic=False trades that guarantee for cudnn's autotuned
  (usually faster) kernels.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = deterministic
  torch.backends.cudnn.benchmark = not deterministic


@dataclass
class TrainConfig:
  epochs: int
  lr: float
  weight_decay: float
  seed: int
  log_every: int = 50
  device: str = "cuda"


@dataclass
class TrainJob:
  model: nn.Module
  loader: DataLoader
  optimizer: Optimizer
  criterion: nn.Module
  config: TrainConfig
  run: Optional[Run] = None
  val_loader: Optional[DataLoader] = None  # if set, evaluated + tracked every epoch


# trainable components of the orchestrator, matched against parameter name
# tokens; query/key norms are the early warning for vanishing gradients since
# they only receive gradient through the attention softmax
_GRAD_COMPONENTS = ("query_head", "key_head", "value_head", "decoder", "fc", "trunk")


def _component_of(param_name: str) -> Optional[str]:
  tokens = param_name.split(".")
  for c in _GRAD_COMPONENTS:
    if c in tokens:
      return c
  return None


def grad_norms(model: nn.Module) -> dict[str, float]:
  """L2 norm of the gradients, total and per component."""
  groups: dict[str, list] = {c: [] for c in _GRAD_COMPONENTS}
  everything = []
  for name, p in model.named_parameters():
    if p.grad is None:
      continue
    everything.append(p)
    c = _component_of(name)
    if c is not None:
      groups[c].append(p)

  def norm(params) -> float:
    if not params:
      return 0.0
    return torch.norm(torch.stack([p.grad.detach().norm(2) for p in params])).item()

  out = {"total": norm(everything)}
  out.update({c: norm(ps) for c, ps in groups.items() if ps})
  return out


def update_to_weight_ratios(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
  """‖Δθ‖/‖θ‖ per component after an optimizer step.

  The 'are the weights actually moving' diagnostic: ~1e-3 per step is healthy,
  ≲1e-6 is effectively frozen, ≳1e-1 is thrashing. Immune to Adam's rescaling.
  """
  num: dict[str, float] = {"total": 0.0}
  den: dict[str, float] = {"total": 0.0}
  for name, p in model.named_parameters():
    old = before.get(name)
    if old is None:
      continue
    d = (p.detach() - old).pow(2).sum().item()
    w = old.pow(2).sum().item()
    for key in ("total", _component_of(name)):
      if key is not None:
        num[key] = num.get(key, 0.0) + d
        den[key] = den.get(key, 0.0) + w
  return {k: num[k] ** 0.5 / max(den[k] ** 0.5, 1e-12) for k in num}


def answer_shift_stats(pre: torch.Tensor, post: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
  """How specialists changed their answers in response to messages.

  pre/post: (n_specialists, batch, classes) logits from before/after
  communication (Orchestrator.last_shift); y: (batch,) labels.

  corrected/corrupted are the flips split by direction — their gap is the per-
  specialist accuracy value of communication. kl catches probability shifts too
  small to flip the argmax. agreement_pre/post guard against homogenization:
  post-agreement rising without corrected − corrupted rising means the
  specialists converge on each other rather than on the truth.
  """
  pre_ans = pre.argmax(-1)    # (n, b)
  post_ans = post.argmax(-1)
  right_pre = pre_ans == y
  right_post = post_ans == y

  kl = (post.softmax(-1) * (F.log_softmax(post, -1) - F.log_softmax(pre, -1))).sum(-1)

  out = {
    "flip_rate": (pre_ans != post_ans).float().mean().item(),
    "corrected": (~right_pre & right_post).float().mean().item(),
    "corrupted": (right_pre & ~right_post).float().mean().item(),
    "kl": kl.mean().item(),
  }

  pairs = [(i, j) for i in range(pre.shape[0]) for j in range(i + 1, pre.shape[0])]
  if pairs:
    out["agreement_pre"] = torch.stack(
      [(pre_ans[i] == pre_ans[j]).float().mean() for i, j in pairs]).mean().item()
    out["agreement_post"] = torch.stack(
      [(post_ans[i] == post_ans[j]).float().mean() for i, j in pairs]).mean().item()
  return out


def _track_answer_shift(run: Run, stats: dict[str, float], prefix: str, step: int, epoch: int) -> None:
  # kl is unbounded while the other kinds live in [0, 1] — give it its own
  # metric name so it never shares a y-axis with the rates
  for kind, v in stats.items():
    if kind == "kl":
      run.track(v, name=f"{prefix}_kl", step=step, epoch=epoch)
    else:
      run.track(v, name=prefix, step=step, epoch=epoch, context={"kind": kind})


@torch.no_grad()
def _evaluate_with_shift(
  model: nn.Module,
  loader: DataLoader,
  device: str,
  criterion: nn.Module,
  **model_kwargs,
) -> tuple[dict[str, float], dict[str, float]]:
  """Single val pass returning (loss/accuracy, answer-shift stats)."""
  model.eval()
  correct = total = 0
  loss_sum = 0.0
  shift_sums: dict[str, float] = {}
  for x, y in loader:
    x, y = x.to(device), y.to(device)
    out = model(x, **model_kwargs)
    correct += (out.argmax(-1) == y).sum().item()
    loss_sum += criterion(out, y).item() * y.size(0)
    total += y.size(0)
    shift = getattr(model, "last_shift", None)
    if shift is not None:
      for k, v in answer_shift_stats(shift["pre"], shift["post"], y).items():
        shift_sums[k] = shift_sums.get(k, 0.0) + v * y.size(0)
  metrics = {"accuracy": correct / total, "loss": loss_sum / total}
  return metrics, {k: v / total for k, v in shift_sums.items()}


def train(job: TrainJob, *args, **kwargs) -> list[float]:
  cfg = job.config
  job.model.to(cfg.device)
  job.model.train()
  losses: list[float] = []
  step = 0

  # endless val-batch stream: one batch per training step gives val_loss the
  # same frequency (and single-batch noise profile) as train_loss at the cost
  # of one forward pass, instead of a 40x-slower full sweep per step
  val_iter = None
  if job.val_loader is not None and job.run is not None:
    def _cycle(loader):
      while True:
        yield from loader
    val_iter = _cycle(job.val_loader)

  pbar = tqdm(total=cfg.epochs * len(job.loader), desc="train")
  try:
    for epoch in range(cfg.epochs):
      correct = seen = 0
      for x, y in job.loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        job.optimizer.zero_grad()

        out = job.model(x, *args, **kwargs)
        loss = job.criterion(out, y)

        correct += (out.detach().argmax(1) == y).sum().item()
        seen += y.size(0)

        loss.backward()
        grads = grad_norms(job.model)
        before = {name: p.detach().clone()
                  for name, p in job.model.named_parameters() if p.requires_grad}
        job.optimizer.step()
        losses.append(loss.item())
        if job.run is not None:
          # epoch goes into track()'s epoch axis, NOT the context — context is
          # part of the sequence identity and would fragment it per epoch
          job.run.track(loss.item(), name="train_loss", step=step, epoch=epoch)
          for component, g in grads.items():
            job.run.track(g, name="grad_norm", step=step, epoch=epoch,
                          context={"component": component})
          for component, r in update_to_weight_ratios(job.model, before).items():
            job.run.track(r, name="update_ratio", step=step, epoch=epoch,
                          context={"component": component})
          # message magnitudes, if the model exposes them (Orchestrator does)
          stats = getattr(job.model, "last_stats", None)
          if stats:
            for part, v in stats.items():
              job.run.track(v, name="msg_norm", step=step, epoch=epoch,
                            context={"part": part})
          # how specialists changed their answers in response to messages
          shift = getattr(job.model, "last_shift", None)
          if shift is not None:
            _track_answer_shift(job.run, answer_shift_stats(shift["pre"], shift["post"], y),
                                "answer_shift", step, epoch)
          # streaming single-batch val loss; must run after the train-batch
          # diagnostics above since this forward overwrites last_stats/last_shift
          if val_iter is not None:
            vx, vy = next(val_iter)
            vx, vy = vx.to(cfg.device), vy.to(cfg.device)
            with torch.no_grad():
              val_batch_loss = job.criterion(job.model(vx, *args, **kwargs), vy).item()
            job.run.track(val_batch_loss, name="val_loss", step=step, epoch=epoch)
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grads['total']:.2e}")
        step += 1

      # accuracies live on an epoch axis (step == epoch) so early stopping can
      # be read off directly; per-step val_loss is handled in the batch loop
      train_acc = correct / max(seen, 1)
      epoch_line = f"epoch {epoch + 1}/{cfg.epochs}  train_accuracy {train_acc:.4f}"
      if job.run is not None:
        job.run.track(train_acc, name="train_accuracy", step=epoch, epoch=epoch)
      if job.val_loader is not None:
        val, val_shift = _evaluate_with_shift(
          job.model, job.val_loader, cfg.device, job.criterion, **kwargs)
        job.model.train()  # the val pass leaves the model in eval mode
        if job.run is not None:
          job.run.track(val["accuracy"], name="val_accuracy", step=epoch, epoch=epoch)
          _track_answer_shift(job.run, val_shift, "val_answer_shift", epoch, epoch)
        epoch_line += "  " + "  ".join(f"val_{k} {v:.4f}" for k, v in val.items())
      pbar.write(epoch_line)
  except KeyboardInterrupt:
    print(f"interrupted at step {step}")
  return losses


# ---------------------------------------------------------------------------
# Data: ensemble split for training, shared val, official test set
# ---------------------------------------------------------------------------

def make_loaders(
  split_file: Path,
  batch_size: int,
  data_root: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
  train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
  ])
  eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
  ])

  split = torch.load(split_file, weights_only=True)
  train_set = datasets.CIFAR100(data_root, train=True, transform=train_tf, download=True)
  val_set = datasets.CIFAR100(data_root, train=True, transform=eval_tf, download=True)
  test_set = datasets.CIFAR100(data_root, train=False, transform=eval_tf, download=True)

  train_loader = DataLoader(
    Subset(train_set, split["ensemble_indices"]),
    batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
  )
  val_loader = DataLoader(
    Subset(val_set, split["val_indices"]),
    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
  )
  test_loader = DataLoader(
    test_set,
    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
  )
  return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Method: communicative (TarMAC-aligned)
# ---------------------------------------------------------------------------

def _load_backbone(path: Path, device: str) -> ResNet:
  ckpt = torch.load(path, map_location=device, weights_only=True)
  # depth comes from the checkpoint so a pool of any CIFAR ResNet loads; older
  # checkpoints predate the field and are all ResNet-20
  depth = ckpt.get("metadata", {}).get("depth", 20)
  model = ResNet(depth=depth).to(device)
  model.load_state_dict(ckpt["state_dict"])
  model.eval()
  return model


@dataclass
class SpecialistSpec:
  """One specialist's construction config: which backbone, which layer it
  reads from/injects into, which head implementations to use.

  early/late and encoder/decoder are legitimately per-specialist — each is
  internal to one specialist's own backbone/heads. key_dim/value_dim are NOT
  here: Orchestrator.forward stacks every specialist's Q/K/V into single
  (b, n, key_dim)/(b, n, value_dim) tensors, so every specialist's encoder
  must project into the same dimensionality — that stays an
  Orchestrator-level setting, passed separately to build_communicative.
  """
  backbone_path: Path
  early: str
  late: str
  encoder: str = "qkv"
  decoder: str = "mlp"


# Swappable head implementations, keyed by the name SpecialistSpec.encoder /
# .decoder refers to. Additive only: a new head architecture is added by
# registering a class here, never by editing build_communicative.
ENCODERS: dict[str, type] = {"qkv": QKVEncoder, "mlp": MLPEncoder}
DECODERS: dict[str, type] = {"mlp": Decoder}


def resolve_specialist_specs(
  backbone_paths: list[Path],
  defaults: argparse.Namespace,
  overrides_path: Optional[Path],
) -> list[SpecialistSpec]:
  """Build one SpecialistSpec per backbone, in order.

  overrides_path is a sparse dict keyed by *position* in backbone_paths (not
  by backbone path — architecture is identical across the pool, so position
  is simpler and needs no index-alignment with a second file), e.g.
  {"1": {"early": "layer1.1"}}. Only positions that differ from the
  CLI-level homogeneous defaults (defaults.early/late) need an entry.
  """
  overrides: dict[str, dict] = {}
  if overrides_path is not None:
    overrides = json.loads(overrides_path.read_text())

  specs = []
  for i, path in enumerate(backbone_paths):
    o = overrides.get(str(i), {})
    specs.append(SpecialistSpec(
      backbone_path=path,
      early=o.get("early", defaults.early),
      late=o.get("late", defaults.late),
      encoder=o.get("encoder", defaults.encoder),
      decoder=o.get("decoder", defaults.decoder),
    ))
  return specs


@torch.no_grad()
def _adapter_dims(backbone: nn.Module, early: str, late: str, device: str) -> tuple[int, int]:
  """Channel widths the decoder injects into / the encoder reads from."""
  shapes = {}
  h_early = backbone.get_submodule(early).register_forward_pre_hook(
    lambda m, inp: shapes.__setitem__("early_in", inp[0].shape)
  )
  h_late = backbone.get_submodule(late).register_forward_hook(
    lambda m, i, out: shapes.__setitem__("late_out", out.shape)
  )
  backbone(torch.zeros(1, 3, 32, 32, device=device))
  h_early.remove()
  h_late.remove()
  return shapes["early_in"][1], shapes["late_out"][1]


def build_communicative(
  specs: list[SpecialistSpec],
  key_dim: int,
  value_dim: int,
  num_classes: int,
  device: str,
) -> Orchestrator:
  from einops.layers.torch import Reduce, Rearrange

  specialists = []
  for spec in specs:
    model = _load_backbone(spec.backbone_path, device)
    # per-spec, not once from specialists[0]: early/late can now differ per
    # specialist, so each one's own adapter dims must be measured separately
    early_ch, late_ch = _adapter_dims(model, spec.early, spec.late, device)
    decoder_cls = DECODERS[spec.decoder]
    encoder_cls = ENCODERS[spec.encoder]
    specialists.append(Specialist(
      Driver(model, spec.early, spec.late),
      decoder_cls(value_dim, early_ch, Rearrange("b c -> b c 1 1")),
      encoder_cls(late_ch, key_dim, value_dim, Reduce("b c h w -> b c", "mean")),
    ))
  return Orchestrator(specialists, key_dim, value_dim, num_classes).to(device)


def _resolve_backbone_paths(manifest: Optional[Path], backbones: Optional[list[Path]]) -> list[Path]:
  """Exactly one of manifest/backbones is set (enforced by an argparse
  mutually-exclusive group), both resolve to the same ordered list."""
  if manifest is not None:
    data = json.loads(manifest.read_text())
    return [Path(p) for p in data["backbones"]]
  return list(backbones)


# ---------------------------------------------------------------------------
# Aggregation across seed replicates
# ---------------------------------------------------------------------------

_SUMMARY_METRICS = ("val_accuracy", "val_loss", "test_accuracy", "test_loss")


def summarize_replicates(experiment: str, seeds: list[int], repo: Optional[str] = None) -> Run:
  """Aggregate a group of N_2 joint-training replicates — same config,
  different --seed, same --experiment (scripts/replicate_joint.py) — into
  one distinguished summary Run in that same experiment: mean/std of each
  final metric across seeds.

  Runs are matched by hparams.seed membership in `seeds`, not by an Aim
  query on a nested field — simpler, and doesn't rely on the query DSL
  reaching into hparams. Every seed must be present: a partial sweep raises
  instead of silently averaging a subset and reporting it as the full group.
  """
  repo_obj = Repo(repo) if repo else Repo.default_repo()
  # A run's hash/experiment are visible to other processes immediately, but
  # its params (hparams, final) and tracked metric sequences are not — Aim
  # only reconciles those into the queryable index on demand (what `aim
  # storage reindex` does), and nothing triggers that automatically for
  # SDK-only writes with no `aim up` server running. Without this, `r["final"]`
  # below raises even for a run that completed and wrote it. Mirrors
  # `aim storage reindex --yes`; private API, but it's what that command
  # itself calls, and a full rebuild over this project's run count is <1s.
  repo_obj._recreate_index()
  candidates = repo_obj.query_runs(f"run.experiment == '{experiment}'").iter_runs()
  matched = [item.run for item in candidates if item.run.get("hparams", {}).get("seed") in seeds]
  if len(matched) != len(seeds):
    found = sorted(r.get("hparams", {}).get("seed") for r in matched)
    raise SystemExit(
      f"expected {len(seeds)} replicate runs for seeds {sorted(seeds)} in "
      f"experiment {experiment!r}, found {len(matched)} (seeds present: {found})"
    )

  summary = Run(experiment=experiment, repo=repo)
  summary.add_tag("summary")
  summary["n_replicates"] = len(matched)
  summary["summarized_seeds"] = sorted(seeds)
  summary["metric_docs"] = metric_docs.summary_as_dict()
  for metric in _SUMMARY_METRICS:
    values = [r["final"][metric] for r in matched]
    summary.track(float(np.mean(values)), name=f"{metric}_mean")
    summary.track(float(np.std(values)), name=f"{metric}_std")
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description="Train an ensemble over frozen backbones")
  sub = parser.add_subparsers(dest="method", required=True)

  p = sub.add_parser("communicative", help="TarMAC-aligned communicative ensemble")
  backbone_group = p.add_mutually_exclusive_group(required=True)
  backbone_group.add_argument("--manifest", type=Path,
                 help="selection/ manifest JSON — trains on its backbones list, in order")
  backbone_group.add_argument("--backbones", type=Path, nargs="+",
                 help="Backbone checkpoint paths (≥2), for ad hoc runs outside selection")
  p.add_argument("--specialist-config", type=Path, default=None,
                 help='JSON {"<index>": {early?, late?, encoder?, decoder?}} sparse '
                      "per-specialist overrides, keyed by position in the resolved "
                      "backbone list — only positions differing from --early/--late "
                      "need an entry")
  p.add_argument("--split-file", type=Path, default=_REPO_ROOT / "splits/three_way_seed42.pt")
  p.add_argument("--early", default="layer1.2", help="Decoder injection layer (default for all specialists)")
  p.add_argument("--late", default="layer3.0", help="Encoder read layer (default for all specialists)")
  p.add_argument("--encoder", default="qkv", choices=sorted(ENCODERS),
                 help="Encoder head architecture (default for all specialists)")
  p.add_argument("--decoder", default="mlp", choices=sorted(DECODERS),
                 help="Decoder head architecture (default for all specialists)")
  p.add_argument("--key-dim", type=int, default=16)
  p.add_argument("--value-dim", type=int, default=64)
  p.add_argument("--k-rounds", type=int, default=1,
                 help="Communication rounds (backbone passes = k_rounds + 1)")
  p.add_argument("--epochs", type=int, default=10)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--weight-decay", type=float, default=1e-4)
  p.add_argument("--batch-size", type=int, default=128)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                 help="cudnn.deterministic (default: on). --no-deterministic trades exact "
                      "reproducibility for cudnn's autotuned kernels.")
  p.add_argument("--num-classes", type=int, default=100)
  p.add_argument("--experiment", default="communicative", help="Aim experiment name")
  p.add_argument("--no-track", action="store_true", help="Disable Aim tracking")
  p.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "ensemble/checkpoints")
  p.add_argument("--data-root", type=str, default=str(_REPO_ROOT / "data"))
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

  ps = sub.add_parser("summarize", help="Aggregate a seed-replicate group into a summary run")
  ps.add_argument("--experiment", required=True, help="Aim experiment name shared by the replicate runs")
  ps.add_argument("--seeds", type=int, nargs="+", required=True, help="Joint seeds to aggregate")
  ps.add_argument("--repo", type=str, default=None, help="Aim repo path (default: Aim's default repo)")

  args = parser.parse_args()

  if args.method == "summarize":
    run = summarize_replicates(args.experiment, args.seeds, repo=args.repo)
    print(f"[summarize] wrote summary run {run.hash} in experiment {args.experiment!r}")
    return

  backbone_paths = _resolve_backbone_paths(args.manifest, args.backbones)
  if len(backbone_paths) < 2:
    parser.error("need at least 2 backbones")

  seed_everything(args.seed, args.deterministic)

  cfg = TrainConfig(
    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
    seed=args.seed, device=args.device,
  )

  train_loader, val_loader, test_loader = make_loaders(
    args.split_file, args.batch_size, args.data_root
  )
  specs = resolve_specialist_specs(backbone_paths, args, args.specialist_config)
  orchestrator = build_communicative(
    specs, args.key_dim, args.value_dim, args.num_classes, args.device,
  )
  n_trainable = sum(p.numel() for p in orchestrator.parameters() if p.requires_grad)
  print(f"[communicative] {len(backbone_paths)} backbones  k_rounds={args.k_rounds}  "
        f"trainable params={n_trainable:,}")

  # the orchestrator readout is log of prob-averaged member softmaxes, so the
  # loss pairs it with NLLLoss (CrossEntropyLoss would double-apply log_softmax)
  criterion = nn.NLLLoss()

  # static reference: the untrained prob-averaging ensemble (k_rounds=0 is a
  # plain mean over the frozen backbones' softmaxes — communication switched off)
  baseline = {}
  for name, loader in (("val", val_loader), ("test", test_loader)):
    res = evaluate(orchestrator, loader, device=args.device, criterion=criterion, k_rounds=0)
    baseline.update({f"{name}_{k}": v for k, v in res.items()})
  print(f"[baseline avg-probs] {baseline}")

  run = None
  if not args.no_track:
    run = Run(experiment=args.experiment)
    run["hparams"] = {
      **asdict(cfg),
      "method": "communicative",
      "backbones": [str(p) for p in backbone_paths],
      "manifest": str(args.manifest) if args.manifest else None,
      "specialist_config": str(args.specialist_config) if args.specialist_config else None,
      "split_file": str(args.split_file),
      "early": args.early, "late": args.late,
      "encoder": args.encoder, "decoder": args.decoder,
      "key_dim": args.key_dim, "value_dim": args.value_dim,
      "k_rounds": args.k_rounds,
      "batch_size": args.batch_size,
      "deterministic": args.deterministic,
    }
    run["baseline_avg_probs"] = baseline
    run["metric_docs"] = metric_docs.as_dict()
    print(f"aim run hash: {run.hash}")

  optimizer = torch.optim.AdamW(
    (p for p in orchestrator.parameters() if p.requires_grad),
    lr=cfg.lr, weight_decay=cfg.weight_decay,
  )
  job = TrainJob(
    model=orchestrator, loader=train_loader, optimizer=optimizer,
    criterion=criterion, config=cfg, run=run, val_loader=val_loader,
  )
  train(job, k_rounds=args.k_rounds)

  results = {}
  for name, loader in (("val", val_loader), ("test", test_loader)):
    results[name] = evaluate(
      orchestrator, loader, device=cfg.device, criterion=criterion, k_rounds=args.k_rounds
    )
    print(f"{name}: {results[name]}")
    if run is not None:
      # final_ prefix keeps these single points out of the per-epoch series
      for k, v in results[name].items():
        run.track(v, name=f"final_{name}_{k}")

  # flat {val_accuracy, val_loss, test_accuracy, test_loss} — also stored as a
  # plain run param (not just .track()'d) so summarize_replicates can read it
  # back with a dict lookup instead of a metric-sequence query
  final_flat = {f"{name}_{k}": v for name, res in results.items() for k, v in res.items()}
  if run is not None:
    run["final"] = final_flat

  stems = "_".join(p.stem for p in backbone_paths)
  out = args.output_dir / f"communicative_{stems}.pt"
  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
    "state_dict": orchestrator.state_dict(),
    "metadata": {
      "method": "communicative",
      "backbones": [str(p) for p in backbone_paths],
      "split_file": str(args.split_file),
      "early": args.early, "late": args.late,
      "encoder": args.encoder, "decoder": args.decoder,
      "key_dim": args.key_dim, "value_dim": args.value_dim,
      "k_rounds": args.k_rounds,
      "baseline_avg_probs": baseline,
      **final_flat,
      **asdict(cfg),
    },
  }, out)
  print(f"saved → {out}")


if __name__ == "__main__":
  main()

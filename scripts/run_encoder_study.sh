#!/usr/bin/env bash
# Encoder architecture study — baseline / wide / mlp arms, one fixed expert
# group (selection/manifests/encoder_study_group.json), replicated over the
# split file's joint_seeds. Run one arm at a time:
#   bash scripts/run_encoder_study.sh baseline
#   bash scripts/run_encoder_study.sh wide
#   bash scripts/run_encoder_study.sh mlp
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_FILE=splits/three_way_seed42.pt
MANIFEST=selection/manifests/encoder_study_group.json
EPOCHS=70
K_ROUNDS=1

arm="${1:-}"
case "$arm" in
  baseline)
    experiment=encoder_baseline
    encoder=qkv; key_dim=16; value_dim=64
    ;;
  wide)
    experiment=encoder_wide
    encoder=qkv; key_dim=64; value_dim=128
    ;;
  mlp)
    experiment=encoder_mlp
    encoder=mlp; key_dim=16; value_dim=64
    ;;
  mlp_wide)
    experiment=encoder_mlp_wide
    encoder=mlp; key_dim=64; value_dim=128
    ;;
  *)
    echo "usage: $0 {baseline|wide|mlp|mlp_wide}" >&2
    exit 1
    ;;
esac

uv run python scripts/replicate_joint.py \
  --split-file "$SPLIT_FILE" \
  --experiment "$experiment" \
  -- \
  communicative \
  --manifest "$MANIFEST" \
  --encoder "$encoder" \
  --key-dim "$key_dim" \
  --value-dim "$value_dim" \
  --epochs "$EPOCHS" \
  --k-rounds "$K_ROUNDS"

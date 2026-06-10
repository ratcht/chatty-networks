import numpy as np


def make_controlled_overlap_stratified_subsets(
    targets,
    frac=0.75,
    overlap_frac=0.60,
    seed=0,
):
    """
    Create two stratified subsets with controlled per-class overlap.

    Args:
        targets: list or array of integer class labels.
        frac: fraction of each class assigned to each model.
        overlap_frac: fraction of each class shared by both models.
        seed: RNG seed.

    Returns:
        indices1, indices2: lists of dataset indices.

    Constraints:
        overlap_frac <= frac
        overlap_frac >= 2 * frac - 1

    The second condition ensures there are enough examples in each class
    without needing replacement.
    """
    if overlap_frac > frac:
        raise ValueError("overlap_frac cannot exceed frac.")

    if overlap_frac < max(0.0, 2.0 * frac - 1.0):
        raise ValueError(
            "overlap_frac is too small for the requested subset fraction. "
            "Need overlap_frac >= 2 * frac - 1."
        )

    targets = np.asarray(targets)
    classes = np.unique(targets)
    rng = np.random.default_rng(seed)

    indices1 = []
    indices2 = []

    for cls in classes:
        cls_indices = np.flatnonzero(targets == cls).copy()
        rng.shuffle(cls_indices)

        n_cls = len(cls_indices)

        n_each = int(round(frac * n_cls))
        n_overlap = int(round(overlap_frac * n_cls))
        n_unique_each = n_each - n_overlap

        needed = n_overlap + 2 * n_unique_each

        if needed > n_cls:
            raise ValueError(
                f"Class {cls}: requested overlap/subset sizes require "
                f"{needed} examples but only {n_cls} are available."
            )

        overlap = cls_indices[:n_overlap]
        unique1 = cls_indices[n_overlap:n_overlap + n_unique_each]
        unique2 = cls_indices[
            n_overlap + n_unique_each:
            n_overlap + 2 * n_unique_each
        ]

        cls_indices1 = np.concatenate([overlap, unique1])
        cls_indices2 = np.concatenate([overlap, unique2])

        indices1.extend(cls_indices1.tolist())
        indices2.extend(cls_indices2.tolist())

    rng.shuffle(indices1)
    rng.shuffle(indices2)

    return indices1, indices2
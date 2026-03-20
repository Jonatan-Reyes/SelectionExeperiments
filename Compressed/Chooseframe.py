from typing import Any, List

import numpy as np
import pandas as pd
from statistics import mode


# Global holder for the state that previously lived on the class instance.
# Assign this from your calling code before using the functions below.
self: Any | None = None


def select_best_model_images() -> List[int]:
    """Select the best model images from intermediate results.

    This is a module-level version of the original class method.
    It expects that the global ``self`` has ``frame_selector_config``
    and ``intermediate_results`` attributes set appropriately.
    """

    expected_cols = self.frame_selector_config.num_classes  # type: ignore[union-attr]
    raw_scores = self.intermediate_results["model_score"]  # type: ignore[union-attr]

    # Filter out rows that don't have exactly num_classes elements
    # (e.g. empty lists from decode errors or frame/score misalignment)
    valid_rows: list[list[float]] = []
    valid_indices: list[int] = []
    for i, row in enumerate(raw_scores):
        if isinstance(row, (list, tuple)) and len(row) == expected_cols:
            valid_rows.append(list(row))
            valid_indices.append(i)

    if not valid_rows:
        print(
            "No valid model scores available for model image selection "
            "(total rows=%d, expected cols=%d) — skipping.",
            len(raw_scores),
            expected_cols,
        )
        return []

    intermediate_results = np.array(valid_rows, dtype=float)
    if (
        intermediate_results.size == 0
        or intermediate_results.ndim < 2
        or intermediate_results.shape[1] == 0
    ):
        print(
            "No model scores available for model image selection "
            "(shape=%s) — skipping.",
            intermediate_results.shape if intermediate_results.ndim >= 1 else "empty",
        )
        return []

    model_prob = np.exp(intermediate_results) / np.sum(
        np.exp(intermediate_results), axis=1, keepdims=True
    )

    if self.frame_selector_config.bad_mes_index >= model_prob.shape[1]:  # type: ignore[union-attr]
        print(
            "bad_mes_index=%d is out of bounds for model_prob with %d columns — skipping deletion.",
            self.frame_selector_config.bad_mes_index,  # type: ignore[union-attr]
            model_prob.shape[1],
        )
        model_prob_no_bad = model_prob
    else:
        model_prob_no_bad = np.delete(
            model_prob,
            self.frame_selector_config.bad_mes_index,
            axis=1,  # type: ignore[union-attr]
        )

    df = pd.DataFrame(intermediate_results)
    df["indices"] = valid_indices
    df["prob"] = np.max(model_prob_no_bad, axis=1)

    df["score"] = np.argmax(model_prob_no_bad, axis=1)
    df["rolled_prob"] = df["prob"].rolling(self.frame_selector_config.roll).mean()  # type: ignore[union-attr]
    df.dropna(inplace=True)

    if len(df) == 0:
        # No suitable pictures
        print(
            "No frames passed the probability threshold of %.2f — skipping model image selection.",
            self.frame_selector_config.prob_thresh,  # type: ignore[union-attr]
        )
        return []

    # Rank according to MES and then probability
    df = df.sort_values(by=["score", "rolled_prob"], ascending=False)

    # Remove frames that are too close to each other
    selected_frames: List[int] = []
    while True:
        selected_frame: int = int(df["indices"].values[0])
        selected_frames.append(selected_frame)
        df = df[
            abs(df["indices"] - selected_frame) > self.frame_selector_config.min_dist
        ]  # type: ignore[union-attr]
        if len(df) == 0:
            break

    return selected_frames


def gen_video_score(intermediate_final_scores: List[int]) -> int:
    """Generate an overall video score from per-frame scores.

    This is a module-level version of the original class method.
    It expects the global ``self`` to have ``frame_selector_config.roll`` set.
    """

    assert self.frame_selector_config.roll <= len(intermediate_final_scores), (  # type: ignore[union-attr]
        "Not enough predictions"
    )
    rolling_res = (
        pd.DataFrame(intermediate_final_scores)
        .rolling(self.frame_selector_config.roll)  # type: ignore[union-attr]
        .apply(lambda x: mode(x))
    )
    rolling_res_clean = rolling_res.dropna()
    max_mes = int(rolling_res_clean.max().item())
    return max_mes

import numpy as np


_LABEL_MAP = {1: 0, 3: 1, 5: 2}


def load_subject(file: str):
    """
    Loads one subject .npy (dict with keys X,y) and returns:
      X: (trials, channels, time, 1) float32
      y: (trials,) int64 in {0,1,2} mapped from {1,3,5}
    """
    data = np.load(file, allow_pickle=True).item()

    X = data["X"].astype(np.float32)  # (trials, channels, time)
    y_raw = np.asarray(data["y"]).astype(int)

    # stable mapping 1,3,5 -> 0,1,2 (do not depend on which labels are present)
    try:
        y = np.vectorize(_LABEL_MAP.__getitem__)(y_raw).astype(np.int64)
    except KeyError as e:
        raise ValueError(f"Unexpected label in {file}: {e}") from e

    # add CNN channel dimension
    X = X[..., np.newaxis]  # (trials, channels, time, 1)

    return X, y
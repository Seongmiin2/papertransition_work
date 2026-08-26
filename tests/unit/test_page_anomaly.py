from __future__ import annotations

import numpy as np

from scan2hwpx.training.page_anomaly import fit_linear_autoencoder


def test_linear_autoencoder_scores_unseen_pattern_higher() -> None:
    train = np.zeros((8, 16), dtype=np.float32)
    for index in range(len(train)):
        train[index, index % 4] = 0.1 + index * 0.001
    model = fit_linear_autoencoder(train, None, rank=3, thumbnail_size=4)

    normal = train[:1]
    anomalous = np.ones((1, 16), dtype=np.float32)

    assert model.score(anomalous)[0] > model.score(normal)[0]

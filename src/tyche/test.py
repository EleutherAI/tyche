from tyche.estimator import VolumeConfig, VolumeEstimator


if __name__ == "__main__":

    cfg = VolumeConfig(
        model_type="mlp",
        model_name={"device": "cuda:7"},
        n_samples=100,  # number of MC samples
        iters=15,
        cutoff=1e-2,  # KL-divergence cutoff (nats)
        cache_mode=None,  # see below
        chunking=False,  # whether to use chunk_and_tokenize
        reduction=None,
        device="cuda:7",
        tol=0.035,
    )

    estimator = VolumeEstimator.from_config(cfg)

    estimator.run()

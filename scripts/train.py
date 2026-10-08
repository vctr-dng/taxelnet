"""Train the deterministic V0 model and write its bundle (see taxelnet.training.run_training).

Usage: uv run python scripts/train.py [--tracking.uri URI] [--train.epochs N] [...]
       uv run python scripts/train.py --help    # every --<section>.<field> override
"""

from taxelnet.config import parse_configs
from taxelnet.training import run_training


def main() -> None:
    cfgs = parse_configs()
    result = run_training(cfgs)
    test = result.metrics["test"]
    print(
        f"test MAE micro {test['micro_mae']:.1f} / macro {test['macro_mae']:.1f} TC "
        f"(best epoch {result.metrics['best_epoch']})\n"
        f"bundle: {result.run_dir}\n"
        f"MLflow run {result.run_id} at {cfgs.tracking.resolved_uri()}"
    )


if __name__ == "__main__":
    main()

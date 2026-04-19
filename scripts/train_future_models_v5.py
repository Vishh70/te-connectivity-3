from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.future_model_pipeline import (  # noqa: E402
    DEFAULT_MAX_ROWS_PER_MACHINE,
    DEFAULT_ROW_STRIDE,
    DEFAULT_SOURCE_PATH,
    METRICS_DIR,
    MODEL_DIR,
    build_horizon_dataset,
    discover_feature_columns,
    train_single_horizon_model,
    write_future_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the six future horizon models.")
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--max-rows-per-machine", type=int, default=DEFAULT_MAX_ROWS_PER_MACHINE)
    parser.add_argument("--row-stride", type=int, default=DEFAULT_ROW_STRIDE)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_columns = discover_feature_columns(args.source_path)

    print("Senior future-model training pipeline")
    print(f"  source: {args.source_path}")
    print(f"  features: {len(feature_columns)}")
    print(f"  models: {args.model_dir}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict] = {}
    summary: dict[str, dict] = {}

    for horizon in ("5m", "10m", "15m", "20m", "25m", "30m"):
        print(f"\nTraining horizon {horizon}...")
        dataset = build_horizon_dataset(
            args.source_path,
            horizon=horizon,
            feature_columns=feature_columns,
            max_rows_per_machine=args.max_rows_per_machine,
            row_stride=args.row_stride,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        artifact = train_single_horizon_model(
            horizon,
            dataset,
            output_dir=args.model_dir,
            metrics_dir=args.metrics_dir,
        )
        artifacts[horizon] = artifact
        summary[horizon] = {
            "rows": dict(dataset.split_counts),
            "threshold": artifact.get("threshold"),
            "validation": artifact.get("metrics", {}).get("validation", {}),
            "test": artifact.get("metrics", {}).get("test", {}),
        }

    manifest = write_future_manifest(artifacts, manifest_path=args.model_dir / "future_model_manifest.json", source_path=args.source_path)
    summary_path = args.metrics_dir / "future_training_summary_v5.json"
    summary_path.write_text(json.dumps({"manifest": manifest, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nManifest written to {args.model_dir / 'future_model_manifest.json'}")
    print(f"Training summary written to {summary_path}")


if __name__ == "__main__":
    main()


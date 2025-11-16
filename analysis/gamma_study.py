import argparse
from pathlib import Path

from analysis.plotting import plot_focal_gamma_losses_from_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot empirical Focal Loss vs probability curves (one line per gamma) from a dataset JSON."
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to dataset-level experiment JSON (e.g. outputs/DRIVE_GAMMA_STUDY.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Unet",
        help="Model name to filter by (e.g. Unet or EncoderDecoder).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="",
        help="Optional optimizer name to filter by (e.g. adamW). If empty, use all.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of probability bins between 0 and 1.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Optional path to save the plot image (e.g. analysis/figures/drive_gamma_loss_unet.png).",
    )

    args = parser.parse_args()

    json_path = Path(args.json)
    save_path = Path(args.save_path) if args.save_path else None
    optimizer = args.optimizer if args.optimizer else None

    plot_focal_gamma_losses_from_json(
        json_path=json_path,
        model=args.model,
        optimizer=optimizer,
        bins=args.bins,
        show=save_path is None,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()



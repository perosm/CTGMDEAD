import json
import argparse
import pathlib
from collections import defaultdict


def main():
    args = _parse_args()
    root_configs_save_dir = args.root_configs_save_dir
    epochs = args.epochs
    for config_directory in sorted(root_configs_save_dir.iterdir()):
        path_to_losses_json = config_directory / "train" / "losses.json"
        task_losses = _read_losses_json(path_to_losses_json)
        print(f"Config directory: {config_directory}")
        max_avg_loss = -1
        task_loss_name_avg_value = defaultdict()
        for task in task_losses.keys():
            loss_name_avg_value = {}
            for loss_name, loss_values in task_losses[task].items():
                loss_name_avg_value[loss_name] = sum(loss_values) / epochs
                if loss_name_avg_value[loss_name] > max_avg_loss:
                    max_avg_loss = loss_name_avg_value[loss_name]

            task_loss_name_avg_value[task] = loss_name_avg_value

        for task in task_losses.keys():
            for loss_name, loss_values in task_losses[task].items():
                print(
                    f"\t{loss_name} -> scale factor: {max_avg_loss / task_loss_name_avg_value[task][loss_name]}"
                )


def _read_losses_json(path_to_losses_json: pathlib.Path) -> dict[str, str]:
    with open(path_to_losses_json, "r") as file:
        return json.load(file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rcsd",
        "--root-configs-save-dir",
        default=pathlib.Path("../run_info/"),
        help="Root directory where all training/evaluation information is stored.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5,
        help="Number of epochs based on which we calculate the scalars for each loss.",
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

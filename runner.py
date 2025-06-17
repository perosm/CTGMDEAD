import pathlib
import argparse
import train

DEFAULT_YAML = "default.yaml"
CONFIGS_INFO_DIR = "configs_info"
CONFIG_NUMBERS_TXT = "config_numbers.txt"
CONFIG_NUMBERS_TRACK_RAN_TXT = "config_numbers_track_ran.txt"

from utils.shared.utils import (
    read_configs_txt_file,
    find_configs_yaml_file,
    load_yaml_file,
    write_configs_txt_file,
    save_yaml_file,
)


def main():
    args = _parse_args()
    flag = True
    configs_ran = []
    while flag:
        config_numbers = read_configs_txt_file(
            args.configs_dir_path / CONFIG_NUMBERS_TXT
        )
        default_yaml = load_yaml_file(args.configs_dir_path / DEFAULT_YAML)
        config_yaml_path = find_configs_yaml_file(
            args.configs_dir_path / CONFIGS_INFO_DIR, config_numbers
        )
        config_yaml = load_yaml_file(config_yaml_path.absolute())
        config_yaml.update({"name": config_yaml_path.stem})
        default_yaml.update(config_yaml)
        yaml_save_path = pathlib.Path(config_yaml["save_path"])
        save_yaml_file(yaml_save_path, config_yaml)

        train.train(config_yaml)

        configs_ran.append(config_numbers.pop(0))
        write_configs_txt_file(
            args.configs_dir_path / CONFIG_NUMBERS_TXT, config_numbers
        )
        write_configs_txt_file(
            args.configs_dir_path / CONFIG_NUMBERS_TRACK_RAN_TXT, configs_ran
        )
        flag = len(config_numbers) != 0

    write_configs_txt_file(args.config_numbers_track_ran, configs_ran)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cdp",
        "--configs-dir-path",
        type=pathlib.Path,
        default=pathlib.Path("./configs"),
        help="Path to config_numbers.txt file in which each row"
        "contains a number of a config which will be ran"
        "from one of the subfolders from /configs/ folder",
    )

    parser.add_argument(
        "-cntr",
        "--config-numbers-track-ran",
        type=pathlib.Path,
        default=pathlib.Path("./configs/config_numbers_track_ran.txt"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

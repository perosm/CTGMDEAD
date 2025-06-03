import pathlib
import argparse
import yaml
import train
import eval

DEFAULT_YAML = "default.yaml"
CONFIGS_INFO_DIR = "configs_info"
CONFIG_NUMBERS_TXT = "config_numbers.txt"
CONFIG_NUMBERS_TRACK_RAN_TXT = "config_numbers_track_ran.txt"


def main():
    args = _parse_args()
    flag = True
    configs_ran = []
    while flag:
        config_numbers = _read_configs_txt_file(
            args.configs_dir_path / CONFIG_NUMBERS_TXT
        )
        default_yaml = _load_yaml_file(args.configs_dir_path / DEFAULT_YAML)
        config_yaml_path = _find_configs_yaml_file(
            args.configs_dir_path / CONFIGS_INFO_DIR, config_numbers
        )
        config_yaml = _load_yaml_file(config_yaml_path.absolute())
        config_yaml.update({"name": config_yaml_path.stem})
        default_yaml.update(config_yaml)

        train.train(config_yaml)

        configs_ran.append(config_numbers.pop(0))
        _write_configs_txt_file(
            args.configs_dir_path / CONFIG_NUMBERS_TRACK_RAN_TXT, configs_ran
        )
        flag = len(config_numbers) != 0

    _write_configs_txt_file(args.config_numbers_track_ran, configs_ran)


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


def _read_configs_txt_file(filepath: pathlib.Path) -> list[str]:
    with open(filepath) as f:
        return f.readlines()


def _find_configs_yaml_file(
    configs_directory: pathlib.Path, config_numbers: list[str]
) -> pathlib.Path:
    wanted_config_number = config_numbers[0]
    if configs_directory.is_dir():
        for yaml_file in configs_directory.iterdir():
            curr_config_number = yaml_file.parts[-1].split("_")[0]
            if wanted_config_number == curr_config_number:
                return yaml_file


def _load_yaml_file(yaml_file: pathlib.Path) -> dict:
    with open(yaml_file.absolute()) as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)


def _write_configs_txt_file(path_to_file: pathlib.Path, lines: list[str]) -> None:
    with open(path_to_file, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()

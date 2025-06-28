import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.shared.enums import TaskEnum
from dataset.nuscenes.NuScenesDataset import NuScenesNuImagesDataset
from dataset.nuscenes import dataset_utils as NuScenesNuImagesUtils


def _remove_samples_with_no_object_gt(mode: str) -> None:
    task_samples_to_remove = defaultdict(list)
    dataset = NuScenesNuImagesDataset(
        version=f"v1.0-{mode}",
        nuscenes_kitti_dataroot=f"./data/nuscenes_kitti/{mode}",
        nuimages_dataroot="./data/nuscenes/nuimages",
        num_samples_train=-1,
        num_samples_val=-1,
        mode=mode,
    )

    for sample_idx, data in enumerate(tqdm(dataset, "Loading samples...")):
        sample_token = dataset.sample_list[sample_idx]

        if sample_token not in dataset.nuscenes_sample_list:
            print(
                "All samples with no object detection gt from NuScenes sample list have been removed! Exiting..."
            )
            break

        if data[TaskEnum.object_detection]["gt_info"].shape[0] == 0:
            print(f"Removing sample with token: {sample_token}")
            # plt.imshow(data[TaskEnum.input].permute(1, 2, 0).numpy())
            # plt.show()
            # plt.pause(
            #     0.1
            # )  # https://stackoverflow.com/questions/22899723/how-to-close-a-python-figure-by-keyboard-input
            # plt.close()
            data.pop("projection_matrix")
            for task in data.keys():
                task_samples_to_remove[task].append(
                    dataset.nuscenes_task_dataroot[task]
                    / f"{sample_token}.{NuScenesNuImagesUtils.TASK_FILE_EXTENSION[task]}"
                )

    print(f"Total samples to remove {len(list(task_samples_to_remove.values())[0])}")
    for task, samples in task_samples_to_remove.items():
        print(f"Removing samples from {task} task!")
        for sample in samples:
            os.remove(sample)


if __name__ == "__main__":
    _remove_samples_with_no_object_gt(mode="train")
    _remove_samples_with_no_object_gt(mode="val")

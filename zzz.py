import os
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pathlib


def load_kitti_annotation(label_path, class_map):
    """
    Parses a KITTI label file and returns boxes and labels tensors.
    class_map: dict mapping KITTI class names to integer labels.
    """
    boxes = []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            cls = parts[0]
            if cls not in class_map:
                continue
            # KITTI bbox: left, top, right, bottom in image coords
            bbox = list(map(float, parts[4:8]))
            boxes.append(bbox)
            labels.append(class_map[cls])
    if len(boxes) == 0:
        # no objects, dummy
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
    return boxes, labels


class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, class_map, device):
        image = Image.open(img_path).convert("RGB")
        self.image = F.to_tensor(image).to(device)
        self.boxes, self.labels = load_kitti_annotation(label_path, class_map)
        self.boxes = self.boxes.to(device)
        self.labels = self.labels.to(device)
        self.device = device

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        target = {"boxes": self.boxes.squeeze(0), "labels": self.labels.squeeze(0)}
        return self.image, target


def overfit_single_image(
    img_path: str,
    label_path: str,
    save_vis: str,
    num_epochs: int = 200,
    lr: float = 1e-3,
    iou_thresh: float = 0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define class map (KITTI classes you want to detect)
    # Adjust or extend as needed
    class_map = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}
    num_classes = max(class_map.values()) + 1  # including background

    # Dataset and DataLoader
    dataset = SingleImageDataset(img_path, label_path, class_map, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load pretrained FPN Faster R-CNN and reset head
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.train()

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Training loop (overfit)
    for epoch in tqdm(range(num_epochs), "Training..."):
        for images, targets in loader:
            targets = {
                "boxes": targets["boxes"].view(-1, 4),
                "labels": targets["labels"].view(-1),
            }
            loss_dict = model(images, [targets])
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {losses.item():.4f}")

    # Switch to eval and visualize
    model.eval()
    with torch.no_grad():
        img, tgt = dataset[0]
        preds = model([img])[0]

    # Prepare image for drawing
    img_uint8 = (img * 255).to(torch.uint8)

    # Draw ground truth (green)
    gt_boxes = tgt["boxes"].cpu().to(torch.int64)
    gt_labels = tgt["labels"].cpu()
    gt_labels_str = [str(int(l.item())) for l in gt_labels]
    vis = draw_bounding_boxes(
        img_uint8, boxes=gt_boxes, labels=gt_labels_str, colors="green", width=3
    )
    # Draw predictions (red)
    pred_boxes = preds["boxes"].cpu().to(torch.int64)
    pred_scores = preds["scores"].cpu()
    pred_labels = preds["labels"].cpu()
    # filter by score threshold
    keep = pred_scores > 0.5
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    pred_labels_str = [
        f"{int(l.item())}:{s:.2f}" for l, s in zip(pred_labels, pred_scores)
    ]
    vis = draw_bounding_boxes(
        vis, boxes=pred_boxes, labels=pred_labels_str, colors="red", width=3
    )
    save_vis: pathlib.Path = pathlib.Path(save_vis)
    save_vis.mkdir(parents=True, exist_ok=True)
    # Save visualization
    plt.figure(figsize=(12, 9))
    plt.imshow(vis.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_vis / "output.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Overfit FPN Faster R-CNN on a single KITTI image"
    )
    parser.add_argument(
        "--img",
        type=str,
        help="Path to KITTI image file",
        default="./data/kitti/input/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Path to KITTI label file",
        default="./data/kitti/object_detection_3d/ground_truth/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000001.txt",
    )
    parser.add_argument(
        "--out", type=str, default="../tmp/", help="Output visualization path"
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()
    overfit_single_image(
        args.img, args.label, args.out, num_epochs=args.epochs, lr=args.lr
    )

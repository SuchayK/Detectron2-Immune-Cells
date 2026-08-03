"""Mask R-CNN instance segmentation of immune cells in lymph node microscopy.

Fine-tunes a COCO-pretrained Mask R-CNN (ResNet-50 FPN) on a COCO-format cell
dataset, then reports COCO AP on the held-out validation split.

    python main.py --data-root /content/cells

The dataset is expected to be laid out as:

    <data-root>/train/  images + json_annotation_train.json
    <data-root>/val/    images + json_annotation_val.json
"""

import argparse
import os

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

TRAIN_DS = "cells_train"
VAL_DS = "cells_val"
BASE_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# Inference threshold for the qualitative overlay. Evaluation deliberately uses a
# much lower threshold -- see EVAL_SCORE_THRESH.
VIS_SCORE_THRESH = 0.7

# COCO AP integrates precision over the full recall range, so the evaluator has to
# see low-confidence detections too. Evaluating at the 0.7 display threshold
# truncates the PR curve and understates AP.
EVAL_SCORE_THRESH = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        default="/content/cells",
        help="dataset root containing train/ and val/ (default: Colab's /content/cells)",
    )
    p.add_argument("--output-dir", default="./output", help="checkpoint + metrics output dir")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--base-lr", type=float, default=0.00025)
    p.add_argument("--ims-per-batch", type=int, default=2)
    p.add_argument("--roi-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--sample-image",
        default=None,
        help="validation image to render predictions for (default: first in the val set)",
    )
    return p.parse_args()


def register_datasets(data_root):
    """Register the train/val splits and return the number of annotated classes.

    The class count is read back from the COCO json rather than hardcoded, so the
    model head always matches the annotations it is trained on.
    """
    register_coco_instances(
        TRAIN_DS,
        {},
        os.path.join(data_root, "train", "json_annotation_train.json"),
        os.path.join(data_root, "train"),
    )
    register_coco_instances(
        VAL_DS,
        {},
        os.path.join(data_root, "val", "json_annotation_val.json"),
        os.path.join(data_root, "val"),
    )
    return len(MetadataCatalog.get(TRAIN_DS).thing_classes)


def build_cfg(args, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_MODEL))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(BASE_MODEL)

    cfg.DATASETS.TRAIN = (TRAIN_DS,)
    cfg.DATASETS.TEST = (VAL_DS,)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []  # no LR decay over this short a schedule

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = args.output_dir
    return cfg


def show(image, path):
    """Write the visualization to disk, and display inline when run under Colab."""
    cv2.imwrite(path, image)
    print(f"wrote {path}")
    try:
        from google.colab.patches import cv2_imshow
    except ImportError:
        return
    cv2_imshow(image)


def main():
    args = parse_args()
    setup_logger()

    num_classes = register_datasets(args.data_root)
    print(f"registered {TRAIN_DS}/{VAL_DS} with {num_classes} class(es)")

    cfg = build_cfg(args, num_classes)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    # --- qualitative: overlay predictions on one validation image ---
    val_metadata = MetadataCatalog.get(VAL_DS)
    sample_path = args.sample_image
    if sample_path is None:
        from detectron2.data import DatasetCatalog

        sample_path = DatasetCatalog.get(VAL_DS)[0]["file_name"]

    image = cv2.imread(sample_path)
    if image is None:
        raise FileNotFoundError(f"could not read sample image: {sample_path}")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = VIS_SCORE_THRESH
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    vis = Visualizer(image[:, :, ::-1], val_metadata, scale=1)
    drawn = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    show(drawn.get_image()[:, :, ::-1], os.path.join(cfg.OUTPUT_DIR, "prediction.png"))

    # --- quantitative: COCO AP over the full validation split ---
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = EVAL_SCORE_THRESH
    eval_predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(VAL_DS, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, VAL_DS)
    print(inference_on_dataset(eval_predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    main()

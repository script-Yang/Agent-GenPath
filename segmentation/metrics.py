"""Semantic overlap and foreground boundary metrics."""

import numpy as np


def summarize(hist):
    tp = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - tp
    iou = tp / (union + 1e-10)
    dice = 2 * tp / (hist.sum(1) + hist.sum(0) + 1e-10)
    return {
        "mIoU": float(iou.mean()),
        "mDice": float(dice.mean()),
        "Pixel_Acc": float(tp.sum() / (hist.sum() + 1e-10)),
        "class_iou": iou.tolist(),
        "class_dice": dice.tolist(),
    }


def aggregate(results):
    keys = ["mIoU", "mDice", "Pixel_Acc", "mHD95", "mASD"]
    summary = {}
    for key in keys:
        values = [r[key] for r in results if key in r and r[key] is not None]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "fold_values": values,
            }
    return summary


class SegmentationMetrics:
    def __init__(self, num_classes, boundary=False):
        self.classes = num_classes
        self.boundary = boundary
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.samples = 0
        self.boundaries = [[] for _ in range(num_classes - 1)]
        self.counts = [
            {"class_id": c, "valid": 0, "both_empty": 0, "prediction_empty": 0, "target_empty": 0}
            for c in range(1, num_classes)
        ]

    def update(self, prediction, target):
        classes = self.classes
        self.hist += np.bincount(
            (classes * target + prediction).ravel(), minlength=classes**2
        ).reshape(classes, classes)
        self.samples += len(prediction)
        if not self.boundary:
            return
        from medpy.metric.binary import hd95, asd

        for predicted, truth in zip(prediction, target):
            for c in range(1, classes):
                pc, yc = predicted == c, truth == c
                present_p, present_y = bool(pc.any()), bool(yc.any())
                if present_p and present_y:
                    self.counts[c - 1]["valid"] += 1
                    self.boundaries[c - 1].append((float(hd95(pc, yc)), float(asd(pc, yc))))
                else:
                    key = (
                        "target_empty"
                        if present_p
                        else "prediction_empty" if present_y else "both_empty"
                    )
                    self.counts[c - 1][key] += 1

    def compute(self):
        if not self.samples:
            raise ValueError("Cannot evaluate an empty dataset")
        result = dict(summarize(self.hist), samples=self.samples)
        if self.boundary:
            means = [np.mean(values, axis=0) if values else None for values in self.boundaries]
            valid = [value for value in means if value is not None]
            result.update(
                mHD95=float(np.mean(valid, axis=0)[0]) if valid else None,
                mASD=float(np.mean(valid, axis=0)[1]) if valid else None,
                class_hd95=[float(value[0]) if value is not None else None for value in means],
                class_asd=[float(value[1]) if value is not None else None for value in means],
                boundary_counts=self.counts,
                boundary_protocol={
                    "units": "pixels",
                    "class_ids": list(range(1, self.classes)),
                    "aggregation": "mean of per-class means over valid image-class pairs",
                    "empty_masks": "excluded; counts reported separately",
                    "connectivity": 1,
                    "hd95": "symmetric",
                    "asd": "prediction to target",
                },
            )
        return result

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("OpenCV (cv2) is required to run this script") from exc
import numpy as np


Box = Dict[str, float]


def load_mask(path: Path, threshold: Optional[int]) -> np.ndarray:
    """加载标签图像并转换为二值掩码"""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to load image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = image.astype(np.uint8)
    if threshold is None:
        binary = (mask > 0).astype(np.uint8) * 255
    else:
        _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary


def extract_rboxes(mask: np.ndarray, min_area: float) -> List[Box]:
    """从二值掩码中提取旋转边界框，使逻辑与modelass一致"""
    binary_map = (mask > 0).astype(np.uint8)
    if binary_map.sum() == 0:
        return []

    contour_input = (binary_map * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        contour_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[Box] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        if w <= 0 or h <= 0:
            continue
        if w < h:
            w, h = h, w
            angle += 90.0
        angle = angle % 180

        area = w * h
        if area < min_area:
            continue

        boxes.append(
            {
                "cx": round(float(cx), 2),
                "cy": round(float(cy), 2),
                "w": round(float(w), 2),
                "h": round(float(h), 2),
                "angle": round(float(angle), 2),
                "area": float(area),
            }
        )

    boxes.sort(key=lambda item: item["area"], reverse=True)
    return boxes


def _rect_from_box(box: Box) -> Sequence[Sequence[float]]:
    return ((box["cx"], box["cy"]), (box["w"], box["h"]), box["angle"])


def rotated_iou(box_a: Box, box_b: Box) -> float:
    """计算两个旋转边界框的IoU（交并比）"""
    rect_a = _rect_from_box(box_a)
    rect_b = _rect_from_box(box_b)
    area_a = box_a["w"] * box_a["h"]
    area_b = box_b["w"] * box_b["h"]
    if area_a <= 0 or area_b <= 0:
        return 0.0
    status, intersection = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if status == cv2.INTERSECT_NONE or intersection is None:
        inter_area = 0.0
    else:
        inter_area = cv2.contourArea(intersection)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def evaluate_boxes(
    preds: Sequence[Box],
    gts: Sequence[Box],
    iou_threshold: float,
) -> Dict[str, float]:
    """评估预测框与真实框的匹配效果，返回精确率、召回率、F1分数"""
    if not preds and not gts:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    matched: set[int] = set()
    tp = 0
    fp = 0
    for pred in preds:
        best_iou = 0.0
        best_idx: Optional[int] = None
        for idx, gt in enumerate(gts):
            if idx in matched:
                continue
            iou = rotated_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx is not None:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1
    fn = len(gts) - len(matched)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def serialize_boxes(boxes: Sequence[Box]) -> List[Dict[str, float]]:
    """序列化旋转框数据（去除面积字段，保留核心参数）"""
    return [
        {
            "cx": box["cx"],
            "cy": box["cy"],
            "w": box["w"],
            "h": box["h"],
            "angle": box["angle"],
        }
        for box in boxes
    ]


def process_single_image(input_path: Path, args) -> Dict[str, object]:
    """处理单张图像，提取旋转框并返回结果字典"""
    mask = load_mask(input_path, args.threshold)
    start = time.perf_counter()
    boxes = extract_rboxes(mask, args.min_area)
    elapsed = time.perf_counter() - start

    # 构造结果字典：file_name改为完整路径（绝对路径）
    result = {
        "file_name": str(input_path.absolute()),  # 保留完整绝对路径
        "class_name": args.class_name,
        "defect_count": len(boxes),
        "rboxes": serialize_boxes(boxes),
        "inference_time_sec": elapsed,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBox extractor for defect masks (compatible with model script)"
    )
    parser.add_argument(
        "--input", required=True, help="path to label image or directory to process"
    )
    parser.add_argument("--gt", help="optional ground-truth label image for evaluation")
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="threshold applied to input images; defaults to non-zero binarization",
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=25.0,
        help="minimum contour area to keep in pixels",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching when computing metrics",
    )
    parser.add_argument(
        "--save_json", required=True, help="path to store detection results (required)"
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="unknown",
        help="class name for the processed images (compatible with model script)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    # 收集所有待处理图像路径（只保留结尾带_t的文件）
    image_paths = []
    if input_path.is_dir():
        # 遍历目录下所有支持的图像格式，仅保留文件名以_t结尾的文件
        for ext in ["*.bmp", "*.png", "*.jpg", "*.jpeg"]:
            for img_path in input_path.glob(ext):
                # 只保留文件名（不含扩展名）以_t结尾的文件
                if img_path.stem.endswith("_t"):
                    image_paths.append(img_path)
        print(
            f"Found {len(image_paths)} valid images (only _t suffix) in directory: {input_path}"
        )
    else:
        # 处理单张图像：只接受以_t结尾的文件
        if not input_path.stem.endswith("_t"):
            raise ValueError(
                f"Single image {input_path.name} does not end with _t, excluded"
            )
        image_paths.append(input_path)
        print(f"Processing valid image: {input_path}")

    # 批量处理图像
    all_results = []
    total_time = 0.0
    total_defects = 0

    for img_path in image_paths:
        result = process_single_image(img_path, args)
        all_results.append(result)

        # 累计统计信息
        total_time += result["inference_time_sec"]
        total_defects += result["defect_count"]

        # 打印单张图像处理结果
        print(
            f"Processed {img_path.name}: {result['defect_count']} defects, "
            f"time: {result['inference_time_sec']:.6f}s"
        )

    # 保存所有结果到JSON文件
    with Path(args.save_json).open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2)

    # 打印整体统计信息
    print(f"\nSummary:")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total defects detected: {total_defects}")
    print(f"Average processing time per image: {total_time / len(image_paths):.6f}s")
    print(f"Results saved to: {args.save_json}")

    # 如果提供了真实标签路径，执行评估
    if args.gt:
        gt_path = Path(args.gt)
        if not gt_path.exists():
            raise FileNotFoundError(f"ground-truth image not found: {gt_path}")
        # 检查gt图是否以_t结尾（按需求保留）
        if not gt_path.stem.endswith("_t"):
            raise ValueError(
                f"Ground-truth image {gt_path.name} does not end with _t, excluded"
            )

        print(f"\nEvaluating against ground truth: {gt_path}")
        gt_mask = load_mask(gt_path, args.threshold)
        gt_boxes = extract_rboxes(gt_mask, args.min_area)

        # 仅支持单张图像的评估（预测与真实标签一一对应）
        if len(image_paths) == 1:
            pred_boxes = all_results[0]["rboxes"]
            # 为预测框添加面积字段（评估函数需要）
            pred_boxes_with_area = [
                {**box, "area": box["w"] * box["h"]} for box in pred_boxes
            ]
            metrics = evaluate_boxes(pred_boxes_with_area, gt_boxes, args.iou_threshold)
            print(
                "Precision={:.3f}, Recall={:.3f}, F1={:.3f}".format(
                    metrics["precision"], metrics["recall"], metrics["f1"]
                )
            )
        else:
            print("Evaluation is only supported for single image input")


if __name__ == "__main__":
    main()

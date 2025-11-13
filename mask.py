import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import os

try:
    import cv2  # 用于图像处理
except ImportError as exc:
    raise ImportError("运行此脚本需要安装OpenCV (cv2)") from exc
import numpy as np


# 定义旋转边界框的数据结构
Box = Dict[str, float]


def load_mask(path: Path, threshold: Optional[int]) -> np.ndarray:
    """加载标签图像并转换为二值掩码"""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    # 转换为灰度图（如果是彩色图）
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = image.astype(np.uint8)
    # 二值化处理
    if threshold is None:
        # 非零像素视为前景
        binary = (mask > 0).astype(np.uint8) * 255
    else:
        # 使用指定阈值进行二值化
        _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary


def extract_rboxes(mask: np.ndarray, min_area: float) -> List[Box]:
    """从二值掩码中提取旋转边界框"""
    working = mask.copy()
    # 查找所有外轮廓
    contours, _ = cv2.findContours(working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    for contour in contours:
        # 计算轮廓面积并过滤小面积轮廓
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        if w <= 0 or h <= 0:
            continue

        # 确保宽大于高，并调整角度（统一格式）
        if w < h:
            w, h = h, w
            angle += 90.0
        angle = angle % 180  # 角度归一化到0-180度

        boxes.append(
            {
                "cx": round(float(cx), 2),  # 中心点x坐标
                "cy": round(float(cy), 2),  # 中心点y坐标
                "w": round(float(w), 2),  # 宽度
                "h": round(float(h), 2),  # 高度
                "angle": round(float(angle), 2),  # 旋转角度
                "area": float(area),  # 轮廓面积
            }
        )
    # 按面积降序排序
    boxes.sort(key=lambda item: item["area"], reverse=True)
    return boxes


def _rect_from_box(box: Box) -> Sequence[Sequence[float]]:
    """从边界框数据结构转换为OpenCV的矩形格式"""
    return ((box["cx"], box["cy"]), (box["w"], box["h"]), box["angle"])


def rotated_iou(box_a: Box, box_b: Box) -> float:
    """计算两个旋转边界框的交并比(IoU)"""
    rect_a = _rect_from_box(box_a)
    rect_b = _rect_from_box(box_b)
    area_a = box_a["w"] * box_a["h"]
    area_b = box_b["w"] * box_b["h"]
    if area_a <= 0 or area_b <= 0:
        return 0.0
    # 计算旋转矩形的交叠区域
    status, intersection = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if status == cv2.INTERSECT_NONE or intersection is None:
        inter_area = 0.0
    else:
        inter_area = cv2.contourArea(intersection)
    # 计算IoU
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def evaluate_boxes(
    preds: Sequence[Box],
    gts: Sequence[Box],
    iou_threshold: float,
) -> Dict[str, float]:
    """评估预测边界框与真实边界框的匹配程度"""
    if not preds and not gts:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    matched: set[int] = set()  # 已匹配的真实框索引
    tp = 0  # 真正例
    fp = 0  # 假正例

    for pred in preds:
        best_iou = 0.0
        best_idx: Optional[int] = None
        # 寻找最佳匹配的真实框
        for idx, gt in enumerate(gts):
            if idx in matched:
                continue
            iou = rotated_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        # 判断是否匹配成功
        if best_iou >= iou_threshold and best_idx is not None:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1

    fn = len(gts) - len(matched)  # 假负例
    # 计算精度、召回率和F1分数
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def serialize_boxes(boxes: Sequence[Box]) -> List[Dict[str, float]]:
    """序列化边界框信息（仅保留必要字段用于输出）"""
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
    """处理单张图像，提取旋转边界框并返回结果"""
    mask = load_mask(input_path, args.threshold)
    # 记录处理时间
    start = time.perf_counter()
    boxes = extract_rboxes(mask, args.min_area)
    elapsed = time.perf_counter() - start

    # 整理结果
    result = {
        "file_name": input_path.name,
        "class_name": args.class_name,
        "defect_count": len(boxes),
        "rboxes": serialize_boxes(boxes),
        "inference_time_sec": elapsed,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用于缺陷掩码的旋转边界框提取器（与模型脚本兼容）"
    )
    parser.add_argument("--input", required=True, help="标签图像路径或待处理目录")
    parser.add_argument("--gt", help="用于评估的可选真实标签图像")
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="应用于输入图像的阈值；默认使用非零二值化",
    )
    parser.add_argument(
        "--min_area", type=float, default=25.0, help="保留轮廓的最小面积（像素）"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="计算指标时用于匹配的IoU阈值"
    )
    parser.add_argument("--save_json", required=True, help="存储检测结果的路径（必填）")
    parser.add_argument(
        "--class_name",
        type=str,
        default="unknown",
        help="处理图像的类别名称（与模型脚本兼容）",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入不存在: {input_path}")

    # 收集所有图像路径
    image_paths = []
    if input_path.is_dir():
        # 处理目录下的所有图像文件
        for ext in ["*.bmp", "*.png", "*.jpg", "*.jpeg"]:
            image_paths.extend(input_path.glob(ext))
        print(f"在目录中找到 {len(image_paths)} 张图像: {input_path}")
    else:
        # 处理单张图像
        image_paths.append(input_path)
        print(f"正在处理图像: {input_path}")

    # 处理所有图像
    all_results = []
    total_time = 0.0
    total_defects = 0

    for img_path in image_paths:
        result = process_single_image(img_path, args)
        all_results.append(result)

        # 累计统计信息
        total_time += result["inference_time_sec"]
        total_defects += result["defect_count"]

        # 输出单张图像结果
        print(
            f"已处理 {img_path.name}: 检测到 {result['defect_count']} 个缺陷，"
            f"耗时: {result['inference_time_sec']:.6f}秒"
        )

    # 保存结果到JSON文件
    with Path(args.save_json).open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2)

    # 输出汇总信息
    print(f"\n汇总信息:")
    print(f"处理图像总数: {len(image_paths)}")
    print(f"检测到的缺陷总数: {total_defects}")
    print(f"单张图像平均处理时间: {total_time / len(image_paths):.6f}秒")
    print(f"结果已保存至: {args.save_json}")

    # 如果提供了真实标签，进行评估
    if args.gt:
        gt_path = Path(args.gt)
        if not gt_path.exists():
            raise FileNotFoundError(f"真实标签图像不存在: {gt_path}")

        print(f"\n正在与真实标签对比评估: {gt_path}")
        gt_mask = load_mask(gt_path, args.threshold)
        gt_boxes = extract_rboxes(gt_mask, args.min_area)

        # 仅支持单张图像的评估
        if len(image_paths) == 1:
            pred_boxes = all_results[0]["rboxes"]
            # 为预测框添加面积信息（评估需要）
            pred_boxes_with_area = [
                {**box, "area": box["w"] * box["h"]} for box in pred_boxes
            ]
            metrics = evaluate_boxes(pred_boxes_with_area, gt_boxes, args.iou_threshold)
            print(
                "精度={:.3f}, 召回率={:.3f}, F1分数={:.3f}".format(
                    metrics["precision"], metrics["recall"], metrics["f1"]
                )
            )
        else:
            print("评估功能仅支持单张图像输入")


if __name__ == "__main__":
    main()

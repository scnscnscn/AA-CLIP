import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2


def normalize_filename(file_name: str) -> str:
    """标准化文件名，移除末尾的_t后缀（如Image_123_t.bmp → Image_123.bmp）"""
    path = Path(file_name)
    if path.stem.endswith("_t.bmp"):
        new_stem = path.stem[:-2]  # 切除末尾的_t
        return str(path.with_stem(new_stem))
    return file_name


def load_results(json_path: Path) -> Dict[str, Dict]:
    """加载JSON格式的检测结果，返回{标准化文件名: 对应结果}的字典"""
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    normalized_results = {}
    for item in results:
        original_name = item["file_name"]
        normalized_name = normalize_filename(original_name)
        normalized_results[normalized_name] = item
    return normalized_results


def rotated_iou(box_a: Dict[str, float], box_b: Dict[str, float]) -> float:
    """计算两个旋转矩形框（RBox）的交并比（IoU）"""
    rect_a = ((box_a["cx"], box_a["cy"]), (box_a["w"], box_a["h"]), box_a["angle"])
    rect_b = ((box_b["cx"], box_b["cy"]), (box_b["w"], box_b["h"]), box_b["angle"])
    area_a = box_a["w"] * box_a["h"]
    area_b = box_b["w"] * box_b["h"]
    if area_a <= 0 or area_b <= 0:
        return 0.0
    status, intersection = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if status == cv2.INTERSECT_NONE or intersection is None:
        return 0.0
    inter_area = cv2.contourArea(intersection)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_boxes(
    pred_boxes: List[Dict], gt_boxes: List[Dict], iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """匹配预测旋转框与标注旋转框，返回匹配数、预测总框数、标注总框数"""
    matched_gt = set()
    matched_count = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = rotated_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx != -1:
            matched_count += 1
            matched_gt.add(best_idx)
    return matched_count, len(pred_boxes), len(gt_boxes)


def main():
    parser = argparse.ArgumentParser(
        description="旋转框检测结果匹配评估工具（忽略_t后缀）"
    )
    parser.add_argument(
        "--model_json", required=True, help="模型预测结果的JSON文件路径"
    )
    parser.add_argument(
        "--mask_json", required=True, help="标注（mask）结果的JSON文件路径"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="IoU匹配阈值（默认0.5）"
    )
    args = parser.parse_args()

    # 加载模型预测结果和标注结果（已标准化文件名，忽略_t后缀）
    model_results = load_results(Path(args.model_json))
    mask_results = load_results(Path(args.mask_json))

    # 找出标准化后文件名相同的文件（忽略_t后缀的影响）
    common_files = set(model_results.keys()) & set(mask_results.keys())
    if not common_files:
        print("未找到模型结果和标注结果的共同文件（已忽略_t后缀），无法评估")
        return
    print(f"找到{len(common_files)}个共同文件（已忽略_t后缀），开始评估...\n")

    # 初始化统计变量
    total_matched = 0
    total_model_defects = 0
    total_mask_defects = 0
    total_files = len(common_files)
    count_mismatch = 0

    # 逐文件评估
    for norm_name in common_files:
        model_res = model_results[norm_name]
        mask_res = mask_results[norm_name]
        # 原始文件名（用于输出显示）
        model_original_name = model_res["file_name"]
        mask_original_name = mask_res["file_name"]

        model_boxes = model_res["rboxes"]
        mask_boxes = mask_res["rboxes"]
        model_count = model_res["defect_count"]
        mask_count = mask_res["defect_count"]

        if model_count != mask_count:
            count_mismatch += 1
            print(
                f"警告：文件 {model_original_name} vs {mask_original_name} - "
                f"模型预测缺陷数：{model_count}，标注缺陷数：{mask_count}（数量不匹配）"
            )

        matched, pred_total, gt_total = match_boxes(
            model_boxes, mask_boxes, args.iou_threshold
        )
        total_matched += matched
        total_model_defects += pred_total
        total_mask_defects += gt_total

    # 计算匹配率
    if total_model_defects == 0 and total_mask_defects == 0:
        match_rate = 1.0
    else:
        match_rate = total_matched / max(total_model_defects, total_mask_defects)

    # 输出结果
    print("\n===== 评估结果汇总 =====")
    print(f"参与评估文件数：{total_files}")
    print(
        f"缺陷数量不匹配文件数：{count_mismatch} ({count_mismatch/total_files*100:.1f}%)"
    )
    print(f"模型预测总缺陷数：{total_model_defects}")
    print(f"标注总缺陷数：{total_mask_defects}")
    print(f"匹配成功的缺陷数：{total_matched}")
    print(f"整体匹配率：{match_rate:.2%} (IoU阈值：{args.iou_threshold})")


if __name__ == "__main__":
    main()
# python evals.py --model_json rbox_results.json --mask_json reseo.json

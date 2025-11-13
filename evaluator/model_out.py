import os
import json
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset
from forward_utils import get_adapted_text_embedding, calculate_similarity_map

import warnings

warnings.filterwarnings("ignore")

# 设置CPU线程数
cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _normalize_score_map(score_map: np.ndarray) -> np.ndarray:
    """对分数图进行归一化处理，将值映射到[0,1]区间"""
    min_val = np.min(score_map)
    max_val = np.max(score_map)
    if max_val - min_val < 1e-6:
        return np.zeros_like(score_map, dtype=np.float32)
    return ((score_map - min_val) / (max_val - min_val)).astype(np.float32)


def _extract_rboxes(
    score_map: np.ndarray, threshold: float, min_area: float
) -> list[dict[str, float]]:
    """从分数图中提取旋转边界框（RBox）"""
    normalized_map = _normalize_score_map(score_map)
    binary_map = (normalized_map >= threshold).astype(np.uint8)

    # 检查二值图是否全为0（无缺陷区域）
    if binary_map.sum() == 0:
        return []

    # 轮廓检测
    contour_input = (binary_map * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        contour_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rboxes: list[dict[str, float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        # 过滤无效和过小的边界框
        if w <= 0 or h <= 0:
            continue
        if w < h:
            w, h = h, w
            angle += 90.0
        angle = angle % 180  # 角度归一化到0-180度
        if w * h < min_area:
            continue

        rboxes.append(
            {
                "cx": round(float(cx), 2),
                "cy": round(float(cy), 2),
                "w": round(float(w), 2),
                "h": round(float(h), 2),
                "angle": round(float(angle), 2),
            }
        )
    rboxes.sort(key=lambda item: item["w"] * item["h"], reverse=True)
    return rboxes


def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "Label",
) -> tuple[np.ndarray, list[str]]:
    """生成测试集的分数图预测结果"""
    preds = []
    file_names = []

    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        file_name = input_data["file_name"]
        file_names.extend(file_name)

        # 提取图像补丁特征
        patch_features, _ = model(image)

        # 计算每个补丁的相似度分数图
        patch_preds = []
        for f in patch_features:
            patch_pred = calculate_similarity_map(
                f, class_text_embeddings, img_size, test=True, domain=dataset
            )
            patch_preds.append(patch_pred)

        # 融合所有补丁的分数图
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        preds.append(patch_preds)

    return np.concatenate(preds, axis=0), file_names


def generate_rbox_results(
    preds: np.ndarray,
    file_names: list[str],
    class_name: str,
    threshold: float,
    min_area: float,
    target_width: int,
    target_height: int,
    img_size: int,
) -> list[dict[str, object]]:
    """基于分数图生成旋转边界框（RBox）检测结果，并缩放到目标尺寸"""
    scale_x = target_width / img_size
    scale_y = target_height / img_size
    results = []
    for score_map, file_name in zip(preds, file_names):
        start = time.perf_counter()
        rboxes = _extract_rboxes(score_map, threshold, min_area)
        elapsed = time.perf_counter() - start
        # 缩放坐标到目标尺寸
        scaled_rboxes = []
        for rbox in rboxes:
            scaled_rbox = {
                "cx": round(rbox["cx"] * scale_x, 2),
                "cy": round(rbox["cy"] * scale_y, 2),
                "w": round(rbox["w"] * scale_x, 2),
                "h": round(rbox["h"] * scale_y, 2),
                "angle": rbox["angle"],  # 角度不变
            }
            scaled_rboxes.append(scaled_rbox)
        results.append(
            {
                "file_name": file_name,
                "class_name": class_name,
                "defect_count": len(scaled_rboxes),
                "rboxes": scaled_rboxes,
                "inference_time_sec": elapsed,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="缺陷检测旋转边界框（RBox）生成")
    # 模型参数
    parser.add_argument(
        "--model_name", type=str, default="ViT-L-14-336", help="CLIP模型名称"
    )
    parser.add_argument("--img_size", type=int, default=518, help="图像输入尺寸")
    parser.add_argument("--relu", action="store_true", help="是否使用ReLU激活")
    # 数据参数
    parser.add_argument("--dataset", type=str, default="Label", help="数据集名称")
    parser.add_argument("--shot", type=int, default=4, help="Few-shot样本数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    # 训练相关
    parser.add_argument("--seed", type=int, default=111, help="随机种子")
    parser.add_argument(
        "--save_path", type=str, default="ckpt/baseline", help="模型权重保存路径"
    )
    parser.add_argument(
        "--rbox_output", type=str, default="rbox_results.json", help="RBox结果输出路径"
    )
    # RBox参数
    parser.add_argument("--rbox_threshold", type=float, default=0.5, help="分数阈值")
    parser.add_argument(
        "--rbox_min_area", type=float, default=50.0, help="边界框最小面积阈值"
    )
    # 目标尺寸参数
    parser.add_argument("--target_width", type=int, default=1288, help="目标图像宽度")
    parser.add_argument("--target_height", type=int, default=815, help="目标图像高度")
    parser.add_argument(
        "--text_adapt_weight", type=float, default=0.1, help="文本适配器权重"
    )
    parser.add_argument(
        "--image_adapt_weight", type=float, default=0.1, help="图像适配器权重"
    )
    parser.add_argument("--text_adapt_until", type=int, default=3, help="文本适配层数")
    parser.add_argument("--image_adapt_until", type=int, default=6, help="图像适配层数")

    args = parser.parse_args()
    setup_seed(args.seed)

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载CLIP基础模型和适配器模型
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)
    model.eval()

    # 加载适配器权重
    text_ckpt = os.path.join(args.save_path, "text_adapter.pth")
    if os.path.exists(text_ckpt):
        model.text_adapter.load_state_dict(torch.load(text_ckpt)["text_adapter"])
        adapt_text = True
    else:
        adapt_text = False

    image_ckpts = sorted(
        [f for f in os.listdir(args.save_path) if f.startswith("image_adapter_")]
    )
    assert len(image_ckpts) > 0, "未找到图像适配器权重文件"
    image_ckpt = os.path.join(args.save_path, image_ckpts[-1])  # 加载最新的权重
    model.image_adapter.load_state_dict(torch.load(image_ckpt)["image_adapter"])
    print(f"加载图像适配器权重: {image_ckpt}")

    # 生成类别文本嵌入
    with torch.no_grad():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(
                clip_model, args.dataset, device
            )

    # 加载测试数据集
    kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
    image_datasets = get_dataset(args.dataset, args.img_size, None, args.shot, "test")

    # 生成RBox检测结果
    all_rbox_results = []
    for class_name, image_dataset in image_datasets.items():
        print(f"\n处理类别: {class_name}")
        test_loader = DataLoader(
            image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )

        # 生成该类别的分数图预测
        with torch.no_grad():
            class_text_emb = text_embeddings[class_name]
            preds, file_names = get_predictions(
                model=model,
                class_text_embeddings=class_text_emb,
                test_loader=test_loader,
                device=device,
                img_size=args.img_size,
                dataset=args.dataset,
            )

        # 生成该类别的RBox结果
        class_rbox_results = generate_rbox_results(
            preds,
            file_names,
            class_name,
            args.rbox_threshold,
            args.rbox_min_area,
            args.target_width,
            args.target_height,
            args.img_size,
        )
        all_rbox_results.extend(class_rbox_results)

        # 打印该类别的每张图像处理结果
        for res in class_rbox_results:
            print(
                f"Processed {os.path.basename(res['file_name'])}: {res['defect_count']} defects, "
                f"time: {res['inference_time_sec']:.6f}s"
            )

        # 输出该类别的统计信息
        total_defects = sum([res["defect_count"] for res in class_rbox_results])
        print(
            f"类别 {class_name}: 测试样本数={len(class_rbox_results)}, 检测缺陷数={total_defects}"
        )

    # 保存结果到JSON文件
    with open(args.rbox_output, "w", encoding="utf-8") as f:
        json.dump(all_rbox_results, f, ensure_ascii=False, indent=2)

    # 计算总处理时间
    total_time = sum([res["inference_time_sec"] for res in all_rbox_results])
    avg_time = total_time / len(all_rbox_results) if all_rbox_results else 0

    print(f"\nRBox结果已保存到: {args.rbox_output}")
    print(f"总测试样本数: {len(all_rbox_results)}")
    print(f"总检测缺陷数: {sum([res['defect_count'] for res in all_rbox_results])}")
    print(f"总处理时间: {total_time:.6f}s")
    print(f"平均每张图像处理时间: {avg_time:.6f}s")


if __name__ == "__main__":
    main()
# python evals.py --model_json rbox_results.json --mask_json reseo.json

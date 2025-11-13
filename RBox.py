import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# -------------------------- 核心配置（请根据你的需求修改这部分） --------------------------
# 1. 标注数据列表（直接粘贴所有你的标注数据，保持原格式）
ANNOTATION_DATA = [
    {
        "file_name": "C:\\Users\\WLQVincent\\Desktop\\AA-CLIP-main\\Image_20240923162245446.bmp",
        "class_name": "Label",
        "defect_count": 5,
        "rboxes": [
            {
                "cx": 2.486 * 215.26,
                "cy": 1.573 * 376.97,
                "w": 2.486 * 88.39,
                "h": 1.573 * 33.12,
                "angle": 60.95,
            },
            {
                "cx": 2.486 * 264.5,
                "cy": 1.573 * 165.0,
                "w": 2.486 * 40.0,
                "h": 1.573 * 31.0,
                "angle": 90.0,
            },
            {
                "cx": 2.486 * 380.75,
                "cy": 1.573 * 160.25,
                "w": 2.486 * 36.06,
                "h": 1.573 * 28.28,
                "angle": 135.0,
            },
            {
                "cx": 2.486 * 82.5,
                "cy": 1.573 * 135.5,
                "w": 2.486 * 89.0,
                "h": 1.573 * 69.0,
                "angle": 90.0,
            },
            {
                "cx": 2.486 * 169.8,
                "cy": 1.573 * 91.9,
                "w": 2.486 * 55.45,
                "h": 1.573 * 53.22,
                "angle": 116.57,
            },
        ],
    },
    # 在这里继续添加你的其他标注数据（保持相同格式，用逗号分隔）
]

# 2. 掩膜图输出目录（自动创建，无需手动新建）
OUTPUT_DIR = "mask_with_rboxes"

# 3. 矩形框样式配置
BOX_COLOR = (255, 255, 255)  # 白色框（BGR格式，如需其他颜色可修改，如(0,255,0)为绿色）
BOX_THICKNESS = 2  # 框的线宽
# ----------------------------------------------------------------------------------------


def draw_rotated_box(image, cx, cy, w, h, angle, color, thickness):
    """绘制单个旋转矩形框"""
    # 角度转换为弧度
    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # 计算矩形四个顶点相对中心的坐标
    half_w, half_h = w / 2, h / 2
    points = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    # 旋转+平移得到绝对坐标
    rotated_points = []
    for x, y in points:
        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta
        rotated_points.append((cx + x_rot, cy + y_rot))

    # 绘制闭合矩形
    pts = np.array(rotated_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    return image


def get_image_size(file_path):
    """获取原始图像的尺寸（确保掩膜图与原始图像尺寸一致）"""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"无法读取图像：{file_path}，请检查文件路径是否正确")
    height, width = img.shape[:2]
    return width, height  # 返回 (宽, 高)


def batch_generate_masks():
    """批量生成带旋转矩形框的掩膜图"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 批量处理每个标注数据
    for data in tqdm(ANNOTATION_DATA, desc="生成掩膜图进度"):
        file_path = data["file_name"]
        rboxes = data["rboxes"]

        # 获取原始图像尺寸，创建对应大小的黑色掩膜
        img_width, img_height = get_image_size(file_path)
        mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # 绘制当前图像的所有旋转矩形框
        for box in rboxes:
            mask = draw_rotated_box(
                mask,
                cx=box["cx"],
                cy=box["cy"],
                w=box["w"],
                h=box["h"],
                angle=box["angle"],
                color=BOX_COLOR,
                thickness=BOX_THICKNESS,
            )

        # 生成输出文件名并保存
        file_name = Path(file_path).name
        output_path = os.path.join(OUTPUT_DIR, f"mask_{file_name}")
        cv2.imwrite(output_path, mask)

    print(f"\n所有掩膜图已保存至：{os.path.abspath(OUTPUT_DIR)}")
    print("提示：掩膜图为黑色背景+白色旋转矩形框，尺寸与原始图像完全一致")


if __name__ == "__main__":
    # 自动安装依赖（如果缺少tqdm）
    try:
        from tqdm import tqdm
    except ImportError:
        print("正在安装进度条依赖（tqdm）...")
        os.system("pip install tqdm -q")
        from tqdm import tqdm

    # 开始批量生成
    try:
        batch_generate_masks()
    except Exception as e:
        print(f"运行出错：{str(e)}")

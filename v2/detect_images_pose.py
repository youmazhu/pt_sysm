import os
from ultralytics import YOLO


def _normalize_classes(model, classes):
    """
    规范化类别参数：支持名称或ID，返回允许的类别ID列表或None。
    """
    if classes is None:
        return None
    names_map = getattr(model, 'names', {})
    name_items = (
        names_map.items() if isinstance(names_map, dict)
        else enumerate(names_map) if isinstance(names_map, (list, tuple))
        else []
    )
    targets = classes if isinstance(classes, (list, tuple)) else [t.strip() for t in str(classes).split(',') if t.strip()]
    id_targets, name_targets = set(), set()
    for t in targets:
        if str(t).isdigit():
            id_targets.add(int(t))
        else:
            name_targets.add(str(t).lower())
    mapped_ids = [cid for cid, cname in name_items if str(cname).strip().lower() in name_targets] if name_targets else []
    allowed_ids = sorted(set(mapped_ids).union(id_targets)) if (mapped_ids or id_targets) else None
    return allowed_ids


def detect_images_pose(
    model_path='yolov8n-pose.pt',
    source_dir='.',
    save_dir='runs/pose',
    conf=0.5,
    iou=0.45,
    classes=None,
    imgsz=640,
    device=None,
):
    """
    姿态检测 - 图片目录批量可视化（叠加关键点骨架）。

    参数说明：
    - model_path: 姿态模型权重，如 yolov8n-pose.pt
    - source_dir: 待检测图片目录
    - save_dir: 结果保存目录（将保存叠加骨架后的图片与预测文本）
    - conf, iou: 置信度与IOU阈值
    - classes: 类别筛选（名称或ID，逗号分隔）
    - imgsz: 推理输入尺寸
    - device: 指定设备（如 0 表示GPU，'cpu' 表示CPU）。不指定自动选择。
    """
    # 加载模型
    model = YOLO(model_path)
    # 检查源目录
    if not os.path.exists(source_dir):
        os.makedirs(source_dir, exist_ok=True)
        print(f'创建图片源目录: {source_dir}\n请将待检测的图片放入该目录后重试。')
        return

    # 规范化类别筛选
    allowed_ids = _normalize_classes(model, classes)

    # 组装预测参数
    predict_kwargs = dict(
        source=source_dir,
        save=True,
        save_txt=True,
        conf=conf,
        iou=iou,
        project=save_dir,
        imgsz=imgsz,
        verbose=False,
    )
    if device is not None:
        predict_kwargs['device'] = device
    if allowed_ids is not None:
        predict_kwargs['classes'] = allowed_ids

    # 执行预测（pose模型将自动叠加关键点与骨架）
    model.predict(**predict_kwargs)
    print(f'姿态图片检测完成，结果已保存到：{save_dir}')


if __name__ == '__main__':
    # 简单示例（按需修改）
    detect_images_pose(
        model_path='yolov8n-pose.pt',
        source_dir='D:/images',
        save_dir='runs/pose',
        conf=0.5,
        iou=0.45,
        classes=None,
        imgsz=640,
    )
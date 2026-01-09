import os
from ultralytics import YOLO

def _normalize_classes(model, classes):
    if classes is None:
        return None
    names_map = getattr(model, 'names', {})
    name_items = names_map.items() if isinstance(names_map, dict) \
        else enumerate(names_map) if isinstance(names_map, (list, tuple)) else []
    targets = classes if isinstance(classes, (list, tuple)) \
        else [t.strip() for t in str(classes).split(',') if t.strip()]
    id_targets, name_targets = set(), set()
    for t in targets:
        if str(t).isdigit():
            id_targets.add(int(t))
        else:
            name_targets.add(str(t).lower())
    mapped_ids = [cid for cid, cname in name_items if str(cname).strip().lower() in name_targets] if name_targets else []
    allowed_ids = sorted(set(mapped_ids).union(id_targets)) if (mapped_ids or id_targets) else None
    return allowed_ids

def detect_images_seg(model_path='yolov8n-seg.pt',
                      source_dir='.',
                      save_dir='runs/segment',
                      conf=0.5,
                      iou=0.45,
                      classes=None,
                      imgsz=640,
                      device=None):
    """
    实例分割-图片目录批量可视化
    """
    model = YOLO(model_path)
    allowed_ids = _normalize_classes(model, classes)
    predict_kwargs = dict(
        source=source_dir,
        save=True,           # 保存叠加掩膜后的图片
        save_txt=True,       # 保存结果文本
        conf=conf,
        iou=iou,
        project=save_dir,
        imgsz=imgsz,
        verbose=False
    )
    if device is not None:
        predict_kwargs['device'] = device
    if allowed_ids is not None:
        predict_kwargs['classes'] = allowed_ids

    model.predict(**predict_kwargs)
    print(f'分割完成，结果已保存到：{save_dir}')
    
if __name__ == '__main__':
    # 示例（按需修改）
    detect_images_seg(
        model_path='yolov8n-seg.pt',
        source_dir='D:/images',
        save_dir='E:/yolov8_bicycle/runs/segment',
        conf=0.5,
        iou=0.45,
        classes=None,
        imgsz=640
    )
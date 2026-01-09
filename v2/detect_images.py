from ultralytics import YOLO
import os
import threading

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

def detect_images(model_path=r'E:\yolov8_bicycle\0402-3.pt',
                  source_dir=r'E:\yolov8_bicycle\images\w_check',
                  save_dir=r'E:\yolov8_bicycle\runs\detect',
                  conf=0.5,
                  iou=0.45,
                  classes=None,
                  result_manager=None,
                  progress_callback=None,
                  cancel_event=None):
    """
    对指定目录下的图片进行目标检测
    Args:
        model_path: YOLOv8模型路径，默认使用yolov8n.pt
        source_dir: 待检测图片所在目录
        save_dir: 检测结果保存目录
    """
    try:
        # 加载模型
        model = YOLO(model_path)
        allowed_ids = _normalize_classes(model, classes)
        # 确保源目录存在
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
            print(f'创建图片源目录: {source_dir}')
            print(f'请将待检测的图片放入 {source_dir} 目录中')
            return

        # 获取图片列表
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]

        if not image_files:
            print(f'在 {source_dir} 目录中没有找到图片文件')
            print('支持的图片格式: PNG, JPG, JPEG, BMP, WEBP')

        
        total = len(image_files)
        print(f'找到 {total} 个图片文件，开始处理...')
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        if cancel_event is None:
            cancel_event = threading.Event()

        for idx, fname in enumerate(image_files, start=1):
            if cancel_event.is_set():
                break

            if callable(progress_callback):
                progress_callback(idx, total, fname)

            image_path = os.path.join(source_dir, fname)
            predict_kwargs = dict(
                source=image_path,
                conf=conf,
                iou=iou,
                save=True,
                save_txt=True,
                project=save_dir,
                name="predict",
                exist_ok=True,
                verbose=False
            )
            if allowed_ids is not None:
                predict_kwargs['classes'] = allowed_ids
            model.predict(**predict_kwargs)
        
        if cancel_event.is_set():
            print('检测已取消')
            return

        if callable(progress_callback) and total > 0:
            progress_callback(total, total, image_files[-1])

        print(f'\n检测完成！结果已保存到 {save_dir} 目录')
        print('\n检测结果包含：')
        print('1. 标注后的图片：显示边界框、类别标签和置信度')
        print('2. 检测结果文本文件：包含每个检测对象的详细信息')
    except Exception as e:
        print(f'发生错误: {str(e)}')

if __name__ == '__main__':
    # 使用默认参数运行检测
    detect_images()

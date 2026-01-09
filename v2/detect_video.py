import cv2
from ultralytics import YOLO
import argparse

from PyQt5.QtCore import QThread, pyqtSignal
import threading

class VideoDetector(QThread):
    frame_processed = pyqtSignal(object)
    detection_finished = pyqtSignal(str)
    detection_result = pyqtSignal(object, str, float)  # 新增信号：图像、类别、置信度
    
    def __init__(self, model_path, video_path, save_dir, conf=0.5, iou=0.45, result_manager=None, allowed_classes=None, target_fps=None):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.save_dir = save_dir
        self.conf = conf
        self.iou = iou
        self.result_manager = result_manager  # 添加结果管理器
        self.target_fps = target_fps  # 新增：保存输出帧率
        self._stop_event = threading.Event()
        self.running = True
        # 记录原始 allowed_classes
        self.allowed_classes_raw = allowed_classes
        # 归一化 allowed_classes：None/字符串/列表，"全部"表示不过滤
        if allowed_classes is None:
            self.allowed_classes = None
        elif isinstance(allowed_classes, str):
            name = allowed_classes.strip()
            self.allowed_classes = None if (not name or name == "全部") else {name.lower()}
        else:
            normalized = {str(c).strip().lower() for c in allowed_classes if str(c).strip()}
            if "全部" in normalized:
                normalized.remove("全部")
            self.allowed_classes = normalized if normalized else None
        # 控制台打印接收到的参数
        print(f'[VideoDetector] params: model="{self.model_path}", video="{self.video_path}", '
              f'save_dir="{self.save_dir}", conf={self.conf}, iou={self.iou}, '
              f'target_fps={self.target_fps}, '
              f'allowed_classes_raw={self.allowed_classes_raw}, normalized={self.allowed_classes}')
        self._last_emit_time_by_class = {}
        self.save_throttle_sec = 0.5  # 每个类别保存节流（秒）
    
    def run(self):
        # 初始化模型和视频流
        try:
            import os
            import time
            import torch
            from collections import deque

            model = YOLO(self.model_path)
            device = 0 if torch.cuda.is_available() else 'cpu'
            cap = cv2.VideoCapture(self.video_path)

            # 获取视频参数
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            output_fps = self.target_fps if self.target_fps is not None else (original_fps if original_fps and original_fps > 0 else 30.0)

            # 准备输出路径与写入器（若需要生成结果文件）
            output_path = os.path.join(self.save_dir, 'result_' + os.path.basename(self.video_path))
            fourcc = cv2.VideoWriter_fourcc(*('m','p','4','v'))
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

            # 更小队列，降低累计延迟
            frame_queue = deque(maxlen=10)
            last_display_time = time.time()
            last_frame_time = 0.0

            # 将允许类别名称映射为模型类别ID（供predict的classes使用）
            cls_filter_ids = None
            names_map = getattr(model, 'names', {})
            if self.allowed_classes is not None:
                name_items = names_map.items() if isinstance(names_map, dict) else enumerate(names_map) if isinstance(names_map, (list, tuple)) else []
                cls_filter_ids = [cid for cid, cname in name_items if str(cname).strip().lower() in self.allowed_classes] or None

            while cap.isOpened() and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()

                # 基于“上次处理时间”的采样，避免不合理跳帧
                if self.target_fps is not None and last_frame_time > 0:
                    target_frame_interval = 1.0 / max(1, self.target_fps)
                    if current_time - last_frame_time < target_frame_interval:
                        continue

                # YOLOv8推理（指定 device 和 imgsz，关闭 verbose）
                predict_kwargs = dict(conf=self.conf, iou=self.iou, imgsz=640, device=device, verbose=False)
                if cls_filter_ids is not None:
                    predict_kwargs['classes'] = cls_filter_ids
                results = model.predict(frame, **predict_kwargs)

                # 可视化结果
                annotated_frame = results[0].plot()

                # 如需更流畅预览，可临时注释下一行写盘
                out.write(annotated_frame)

                # 入队（较小maxlen，减少累计延迟）
                frame_queue.append(annotated_frame)
                last_frame_time = current_time

                # 控制显示节奏
                display_interval = 1.0 / output_fps if output_fps and output_fps > 0 else 1.0 / 30
                if current_time - last_display_time >= display_interval and frame_queue:
                    display_frame = frame_queue.popleft()
                    self.frame_processed.emit(display_frame)
                    last_display_time = current_time

                # 结果保存节流：每帧每类别至多保存一次，且每类至少间隔save_throttle_sec
                if self.result_manager and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    classes_seen = set()
                    for box in results[0].boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        class_name = results[0].names[cls_id]
                        if conf >= self.conf:
                            if (self.allowed_classes is None or class_name.lower() in self.allowed_classes) and class_name not in classes_seen:
                                last_emit = self._last_emit_time_by_class.get(class_name, 0.0)
                                if current_time - last_emit >= self.save_throttle_sec:
                                    self._last_emit_time_by_class[class_name] = current_time
                                    self.detection_result.emit(annotated_frame, class_name, conf)
                                    classes_seen.add(class_name)

                if self._stop_event.is_set():
                    break

            # 清尾帧（可选）
            while frame_queue:
                display_frame = frame_queue.popleft()
                self.frame_processed.emit(display_frame)
                time.sleep(1.0 / output_fps if output_fps and output_fps > 0 else 1.0 / 30)

            cap.release()
            out.release()
            try:
                if hasattr(cv2, 'destroyAllWindows'):
                    cv2.destroyAllWindows()
            except Exception:
                pass

            if self._stop_event.is_set():
                self.detection_finished.emit(f'检测已停止，部分结果已保存至 {output_path}')
            else:
                self.detection_finished.emit(f'检测完成，结果已保存至 {output_path}')

        except Exception as e:
            self.detection_finished.emit(f'检测出错: {str(e)}')
        finally:
            self.running = False
    
    def stop(self):
        self._stop_event.set()
        self.running = False
        self.wait(3000)  # 等待线程结束，最多3秒

def detect_video(model_path, video_path, save_dir, conf=0.5, iou=0.45, result_manager=None, allowed_classes=None, target_fps=None):
    """
    视频检测函数
    :param model_path: 模型路径
    :param video_path: 视频文件路径
    :param save_dir: 结果保存目录
    :param conf: 置信度阈值
    :param iou: IOU阈值
    :param result_manager: 检测结果管理器
    :param allowed_classes: 允许的类别（名称或None）
    :param target_fps: 目标输出帧率（None表示沿用源视频FPS或默认30）
    """
    detector = VideoDetector(model_path, video_path, save_dir, conf, iou, result_manager, allowed_classes, target_fps)
    detector.start()
    return detector

# 命令行接口部分
if __name__ == '__main__':
    # 初始化参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input.mp4', help='输入视频路径')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    parser.add_argument('--model', type=str, default='best.pt', help='模型路径')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    # 新增：类别筛选（逗号分隔，支持名称或ID）
    parser.add_argument('--classes', type=str, default='', help='类别筛选（逗号分隔，支持名称或ID，如 "fire,smoke" 或 "0,2"）')
    args = parser.parse_args()

    # 初始化模型和视频流
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.input)

    # 获取视频参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 计算允许的类别ID
    allowed_ids = None
    if args.classes:
        targets = [t.strip() for t in args.classes.split(',') if t.strip()]
        names_map = getattr(model, 'names', {})
        # 兼容 dict 或 list
        name_items = names_map.items() if isinstance(names_map, dict) else enumerate(names_map) if isinstance(names_map, (list, tuple)) else []
        name_targets, id_targets = set(), set()
        for t in targets:
            if t.isdigit():
                id_targets.add(int(t))
            else:
                name_targets.add(t.lower())
        mapped_ids = [cid for cid, cname in name_items if str(cname).strip().lower() in name_targets] if name_targets else []
        allowed_ids = sorted(set(mapped_ids).union(id_targets)) if (mapped_ids or id_targets) else None
        print(f'[CLI] classes_raw="{args.classes}", allowed_ids={allowed_ids}, names_map={names_map}')

    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8推理（带类别筛选）
        predict_kwargs = dict(conf=args.conf)
        if allowed_ids is not None:
            predict_kwargs['classes'] = allowed_ids
        results = model.predict(frame, **predict_kwargs)
        
        # 可视化结果
        annotated_frame = results[0].plot()
        
        # 写入输出视频
        out.write(annotated_frame)
        
        # 实时显示（可选）
        cv2.imshow('Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    try:
        if hasattr(cv2, 'destroyAllWindows'):
            cv2.destroyAllWindows()
    except Exception:
        pass
    print(f'处理完成，结果已保存至 {args.output}')
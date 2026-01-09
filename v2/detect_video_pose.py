import cv2
import time
import threading
import argparse
import os
import torch
from collections import deque
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal


def _normalize_classes(model, classes):
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


class PoseVideoDetector(QThread):
    """
    姿态检测视频线程：实时推理与关键点骨架可视化，支持类别筛选与FPS节流。
    """
    frame_processed = pyqtSignal(object)
    detection_finished = pyqtSignal(str)
    detection_result = pyqtSignal(object, str, float)

    def __init__(self, model_path, video_path, save_dir, conf=0.5, iou=0.45, classes=None, target_fps=None, imgsz=640):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.save_dir = save_dir
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.target_fps = target_fps
        self.imgsz = imgsz
        self._stop_event = threading.Event()
        self.running = True
        self.allowed_ids = None
        self._last_emit_time_by_class = {}
        self.save_throttle_sec = 0.5

    def run(self):
        try:
            model = YOLO(self.model_path)
            # 运行时规范化类别筛选
            self.allowed_ids = _normalize_classes(model, self.classes)
            device = 0 if torch.cuda.is_available() else 'cpu'
            cap = cv2.VideoCapture(self.video_path)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            output_fps = self.target_fps if self.target_fps is not None else (original_fps if original_fps and original_fps > 0 else 30.0)

            output_path = os.path.join(self.save_dir, 'pose_' + os.path.basename(self.video_path))
            fourcc = cv2.VideoWriter_fourcc(*('m','p','4','v'))
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

            frame_queue = deque(maxlen=10)
            last_display_time = time.time()
            last_frame_time = 0.0

            names_map = getattr(model, 'names', {})

            while cap.isOpened() and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()
                if self.target_fps is not None and last_frame_time > 0:
                    interval = 1.0 / max(1, self.target_fps)
                    if current_time - last_frame_time < interval:
                        continue

                predict_kwargs = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=device, verbose=False)
                if self.allowed_ids is not None:
                    predict_kwargs['classes'] = self.allowed_ids
                results = model.predict(frame, **predict_kwargs)

                # 自动叠加关键点与骨架
                annotated = results[0].plot()
                out.write(annotated)
                frame_queue.append(annotated)
                last_frame_time = current_time

                # 控制UI显示帧率
                display_interval = 1.0 / output_fps if output_fps and output_fps > 0 else 1.0 / 30
                if current_time - last_display_time >= display_interval and frame_queue:
                    self.frame_processed.emit(frame_queue.popleft())
                    last_display_time = current_time

                # 通过boxes获取类别与置信度（pose任务通常也包含boxes）
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    classes_seen = set()
                    for box in results[0].boxes:
                        cid = int(box.cls.item())
                        conf = float(box.conf.item())
                        cname = names_map[cid] if isinstance(names_map, dict) else names_map[cid]
                        if conf >= self.conf and cname not in classes_seen:
                            last_emit = self._last_emit_time_by_class.get(cname, 0.0)
                            if current_time - last_emit >= self.save_throttle_sec:
                                self._last_emit_time_by_class[cname] = current_time
                                self.detection_result.emit(annotated, cname, conf)
                                classes_seen.add(cname)

            # 清尾帧
            while frame_queue:
                self.frame_processed.emit(frame_queue.popleft())
                time.sleep(1.0 / output_fps if output_fps and output_fps > 0 else 1.0 / 30)

            cap.release()
            out.release()
            self.detection_finished.emit(f'姿态视频完成，结果已保存至 {output_path}')
        except Exception as e:
            self.detection_finished.emit(f'姿态视频出错: {str(e)}')
        finally:
            self.running = False

    def stop(self):
        self._stop_event.set()
        self.running = False
        self.wait(3000)


def detect_video_pose(model_path, video_path, save_dir, conf=0.5, iou=0.45, classes=None, target_fps=None, imgsz=640):
    det = PoseVideoDetector(model_path, video_path, save_dir, conf, iou, classes, target_fps, imgsz)
    det.start()
    return det


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input.mp4', help='输入视频路径')
    parser.add_argument('--output_dir', type=str, default='runs/pose', help='输出目录')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt', help='姿态模型路径')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--classes', type=str, default='', help='类别筛选，如 "person" 或 "0"')
    parser.add_argument('--fps', type=int, default=None, help='目标输出帧率')
    parser.add_argument('--imgsz', type=int, default=640, help='推理输入尺寸')
    args = parser.parse_args()

    classes = [t.strip() for t in args.classes.split(',')] if args.classes else None
    det = detect_video_pose(args.model, args.input, args.output_dir, args.conf, args.iou, classes, args.fps, args.imgsz)
    det.wait()
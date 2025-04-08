from ultralytics import YOLO
import cv2
import time
import threading
import numpy as np
from typing import Optional
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker
from collections import deque
class StreamDetector(QObject):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, model_path: str, stream_url: str, save_dir: str, conf: float = 0.5, iou: float = 0.45):
        super().__init__()
        self.frame_queue = deque(maxlen=25)
        self.queue_lock = QMutex()
        self.model_path = model_path
        self.stream_url = stream_url
        self.save_dir = save_dir
        self.conf = conf
        self.iou = iou
        self.cap = None
        self.model = None
        self.last_frame_time = 0
        self.frame_interval = 1/30  # 30fps
        self._stop_event = threading.Event()
        self.running = True
        
    def connect_stream(self) -> bool:
        try:
            # 先释放已有资源
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                print(f"无法打开视频流: {self.stream_url}")
                self.cap = None  # 确保释放无效的cap对象
                return False
            
            # 单独初始化模型
            try:
                self.model = YOLO(self.model_path)
            except Exception as model_error:
                print(f"模型加载失败: {str(model_error)}")
                self.cap.release()
                self.cap = None
                return False
            
            self.last_frame_time = time.time()
            return True
        except Exception as e:
            print(f"连接流媒体失败: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def process_stream(self):
        while getattr(self, 'running', True) and not self._stop_event.is_set():
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                time.sleep(0.001)
                continue

            # 前置检查视频流状态
            if self.cap is None or not self.cap.isOpened():
                if not self.reconnect():
                    print("视频流无法恢复，终止检测")
                    break
                continue
                
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("视频流中断，尝试重连...")
                    if not self.reconnect():
                        break
                    continue

                # 跳过解码错误的帧
                if frame is None or frame.size == 0:
                    continue

                results = self.model.track(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    persist=True,
                    save=False,
                    save_txt=False,
                    project=self.save_dir
                )
                
                # 获取原始帧的尺寸
                height, width = frame.shape[:2]
                
                # 在原始帧上添加类型识别正确率信息
                if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        # 创建类别统计字典
                        class_counts = {}
                        for box in boxes:
                            cls_id = int(box.cls.item())
                            conf = float(box.conf.item())
                            class_name = results[0].names[cls_id]
                            if class_name not in class_counts:
                                class_counts[class_name] = {'count': 0, 'total_conf': 0}
                            class_counts[class_name]['count'] += 1
                            class_counts[class_name]['total_conf'] += conf
                        
                        # 在帧上绘制类别统计信息
                        y_offset = 30
                        for cls_name, stats in class_counts.items():
                            avg_conf = stats['total_conf'] / stats['count']
                            text = f"{cls_name}: {stats['count']}个, 平均置信度: {avg_conf:.2f}"
                            cv2.putText(frame, text, (10, y_offset), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            y_offset += 30
                
                # 使用模型的plot方法绘制检测结果
                plotted_frame = results[0].plot()
                plotted_frame = cv2.cvtColor(plotted_frame, cv2.COLOR_BGR2RGB)
                
                with QMutexLocker(self.queue_lock):
                    self.frame_queue.append(plotted_frame)
                self.frame_processed.emit(plotted_frame)
                self.last_frame_time = current_time

            except (cv2.error, ConnectionResetError, RuntimeError) as e:
                print(f"处理错误: {str(e)}")
                if not self.reconnect():
                    break
                # Skip corrupted frames
                time.sleep(0.1)
                    
    def stop(self):
        with QMutexLocker(self.queue_lock):
            self.running = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'writer'):
                self.writer.release()
            self.model = None
            self._stop_event.set()

    def reconnect(self, max_retries=3) -> bool:
        for _ in range(max_retries):
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            time.sleep(1)
            if self.connect_stream():
                print("流重新连接成功")
                return True
        print("流重新连接失败")
        return False


def detect_stream(model_path: str, stream_url: str, save_dir: str, conf: float = 0.5, iou: float = 0.45):
    from PyQt5.QtCore import QThread
    
    class StreamThread(QThread):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector
            
        def run(self):
            self.detector.process_stream()
            
        def stop(self):
            self.detector.stop()
            self.quit()
            self.wait(1000)
    
    detector = StreamDetector(model_path, stream_url, save_dir, conf, iou)
    if detector.connect_stream():
        print(f"成功连接视频流: {stream_url}")
        thread = StreamThread(detector)
        thread.start()
        return thread
    else:
        print("流媒体检测初始化失败")
        return None
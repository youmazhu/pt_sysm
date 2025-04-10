import cv2
from ultralytics import YOLO
import argparse

from PyQt5.QtCore import QThread, pyqtSignal
import threading

class VideoDetector(QThread):
    frame_processed = pyqtSignal(object)
    detection_finished = pyqtSignal(str)
    detection_result = pyqtSignal(object, str, float)  # 新增信号：图像、类别、置信度
    
    def __init__(self, model_path, video_path, save_dir, conf=0.5, iou=0.45, result_manager=None):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.save_dir = save_dir
        self.conf = conf
        self.iou = iou
        self.result_manager = result_manager  # 添加结果管理器
        self._stop_event = threading.Event()
        self.running = True
        
    def run(self):
        # 初始化模型和视频流
        try:
            import os
            model = YOLO(self.model_path)
            cap = cv2.VideoCapture(self.video_path)

            # 获取视频参数
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 准备输出路径
            output_path = os.path.join(self.save_dir, 'result_' + os.path.basename(self.video_path))

            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 逐帧处理
            while cap.isOpened() and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLOv8推理
                results = model.predict(frame, conf=self.conf, iou=self.iou)
                
                # 可视化结果
                annotated_frame = results[0].plot()
                
                # 写入输出视频
                out.write(annotated_frame)
                
                # 发送信号更新UI
                self.frame_processed.emit(annotated_frame)
                
                # 保存检测结果到结果管理器
                if self.result_manager and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        class_name = results[0].names[cls_id]
                        
                        # 只保存置信度高于阈值的检测结果
                        if conf >= self.conf:
                            # 发送检测结果信号
                            self.detection_result.emit(annotated_frame.copy(), class_name, conf)
                
                # 检查是否需要停止
                if self._stop_event.is_set():
                    break

            # 释放资源
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
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

def detect_video(model_path, video_path, save_dir, conf=0.5, iou=0.45, result_manager=None):
    """
    视频检测函数
    :param model_path: 模型路径
    :param video_path: 视频文件路径
    :param save_dir: 结果保存目录
    :param conf: 置信度阈值
    :param iou: IOU阈值
    :param result_manager: 检测结果管理器
    :return: VideoDetector实例
    """
    detector = VideoDetector(model_path, video_path, save_dir, conf, iou, result_manager)
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

    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8推理
        results = model.predict(frame, conf=args.conf)
        
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
    cv2.destroyAllWindows()
    print(f'处理完成，结果已保存至 {args.output}')
import sys
import cv2
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QFileDialog, QWidget, QLabel, QLineEdit, QTextEdit, QComboBox, QSlider, QHBoxLayout, QGridLayout)
from PyQt5.QtGui import QIcon, QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker, QTimer
from detect_images import detect_images
from detect_video import detect_video
from detect_stream import detect_stream

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_style()
        self.statusBar().showMessage('就绪')  # 初始化状态栏
        self.initUI()

    def setup_style(self):
        self.setStyleSheet('''
            QMainWindow {
                background-color: #1E1E2E;
                color: #CDD6F4;
            }
            QPushButton {
                background-color: #45475A;
                color: #CDD6F4;
                border: none;
                qproperty-iconSize: 20px;
                border-radius: 6px;
                padding: 8px;
                min-width: 100px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #585B70;
            }
            QPushButton:pressed {
                background-color: #313244;
            }
            QLineEdit, QTextEdit {
                background-color: #313244;
                border: 1px solid #45475A;
                padding: 8px;
                border-radius: 6px;
                color: #CDD6F4;
            }
            QComboBox {
                background-color: #313244;
                border: 1px solid #45475A;
                padding: 8px;
                border-radius: 6px;
                color: #CDD6F4;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(E:\yolov8_bicycle\icons\down-arrow.svg);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #313244;
                border: 1px solid #45475A;
                selection-background-color: #585B70;
                selection-color: #CDD6F4;
            }
            QLabel {
                color: #BAC2DE;
                font-weight: 500;
            }
            QGroupBox {
                border: 1px solid #45475A;
                border-radius: 6px;
                margin-top: 1.5em;
                padding-top: 0.5em;
                font-weight: bold;
                color: #CDD6F4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #45475A;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #89B4FA;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #89B4FA;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #313244;
                border: 1px solid #45475A;
                color: #CDD6F4;
            }
            QStatusBar {
                background-color: #313244;
                color: #BAC2DE;
            }
        ''')
        
    def initUI(self):
        self.setWindowTitle('权重检测系统')
        self.setGeometry(100, 100, 1200, 800)

        # 创建主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QGroupBox, QHBoxLayout
        main_layout = QHBoxLayout()
        
        # 创建左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # 模型选择分组
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        
        # 模型选择水平布局
        model_path_layout = QHBoxLayout()
        model_path_layout.setContentsMargins(0, 0, 0, 0)
        self.model_path = QLineEdit(r'E:\yolov8_bicycle\0402-3.pt')
        btn_model = QPushButton(QIcon(r'E:\yolov8_bicycle\icons\folder.svg'), '选择模型')
        btn_model.clicked.connect(self.select_model)
        model_path_layout.addWidget(QLabel('模型路径:'))
        model_path_layout.addWidget(self.model_path, 1)
        model_path_layout.addWidget(btn_model)
        model_layout.addLayout(model_path_layout)
        
        # 检测模式选择
        mode_layout = QHBoxLayout()
        self.mode = 'image'  # 默认图片模式
        self.mode_label = QLabel('检测模式:')
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['图片模式', '视频模式', '流媒体模式'])
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_combo, 1)
        model_layout.addLayout(mode_layout)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # 输入源分组
        input_group = QGroupBox("输入设置")
        input_layout = QVBoxLayout()
        
        # 图片目录选择
        img_layout = QHBoxLayout()
        self.img_label = QLabel('图片目录:')
        self.source_dir = QLineEdit(r'E:\yolov8_bicycle\images\w_check')
        self.btn_source = QPushButton(QIcon(r'E:\yolov8_bicycle\icons\folder.svg'), '浏览')
        self.btn_source.clicked.connect(self.select_source_dir)
        img_layout.addWidget(self.img_label)
        img_layout.addWidget(self.source_dir, 1)
        img_layout.addWidget(self.btn_source)
        input_layout.addLayout(img_layout)
        
        # 视频文件选择
        video_layout = QHBoxLayout()
        self.video_file_label = QLabel('视频文件:')  # 重命名为video_file_label以避免命名冲突
        self.video_path = QLineEdit()
        self.btn_video = QPushButton(QIcon(r'E:\yolov8_bicycle\icons\folder.svg'), '浏览')
        self.btn_video.clicked.connect(self.select_video_file)
        video_layout.addWidget(self.video_file_label)
        video_layout.addWidget(self.video_path, 1)
        video_layout.addWidget(self.btn_video)
        input_layout.addLayout(video_layout)
        self.video_path.setVisible(False)
        self.btn_video.setVisible(False)
        self.video_file_label.setVisible(False)
        
        # 流媒体地址输入
        stream_layout = QHBoxLayout()
        self.stream_label = QLabel('流媒体地址:')
        self.stream_url = QLineEdit('rtsp://')
        stream_layout.addWidget(self.stream_label)
        stream_layout.addWidget(self.stream_url, 1)
        input_layout.addLayout(stream_layout)
        
        stream_btn_layout = QHBoxLayout()
        self.btn_stream = QPushButton('输入地址')
        self.btn_test_stream = QPushButton('测试连接')
        self.btn_test_stream.clicked.connect(self.test_stream_connection)
        stream_btn_layout.addWidget(self.btn_stream)
        stream_btn_layout.addWidget(self.btn_test_stream)
        input_layout.addLayout(stream_btn_layout)
        
        self.stream_url.setVisible(False)
        self.btn_stream.setVisible(False)
        self.btn_test_stream.setVisible(False)
        self.stream_label.setVisible(False)
        
        # 输出目录选择
        output_layout = QHBoxLayout()
        output_label = QLabel('输出目录:')
        self.save_dir = QLineEdit(r'E:\yolov8_bicycle\runs\detect')
        btn_save = QPushButton(QIcon(r'E:\yolov8_bicycle\icons\folder.svg'), '浏览')
        btn_save.clicked.connect(self.select_save_dir)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.save_dir, 1)
        output_layout.addWidget(btn_save)
        input_layout.addLayout(output_layout)
        
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
        # 参数设置分组
        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout()
        
        # 置信度阈值滑块
        conf_layout = QVBoxLayout()
        self.conf_label = QLabel('置信度阈值: 0.5')
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        conf_layout.addWidget(self.conf_label)
        conf_layout.addWidget(self.conf_slider)
        param_layout.addLayout(conf_layout)
        
        # IOU阈值滑块
        iou_layout = QVBoxLayout()
        self.iou_label = QLabel('IOU阈值: 0.45')
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        iou_layout.addWidget(self.iou_label)
        iou_layout.addWidget(self.iou_slider)
        param_layout.addLayout(iou_layout)
        
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)
        
        # 操作按钮
        action_layout = QHBoxLayout()
        self.btn_start = QPushButton(QIcon(r'E:\yolov8_bicycle\icons\play.svg'), '开始检测')
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self.start_detection)
        action_layout.addWidget(self.btn_start)
        
        # 添加停止按钮（初始不可见）
        self.btn_stop = QPushButton(QIcon(r'E:\yolov8_bicycle\icons\play.svg'), '停止检测')
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setVisible(False)
        action_layout.addWidget(self.btn_stop)
        
        # 添加断开流媒体按钮（初始不可见）
        self.btn_disconnect_stream = QPushButton('断开连接')
        self.btn_disconnect_stream.setMinimumHeight(40)
        self.btn_disconnect_stream.clicked.connect(self.disconnect_stream)
        self.btn_disconnect_stream.setVisible(False)
        action_layout.addWidget(self.btn_disconnect_stream)
        
        left_layout.addLayout(action_layout)
        
        # 日志显示
        log_group = QGroupBox("操作日志")
        log_layout = QVBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(150)
        log_layout.addWidget(self.log)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        # 创建右侧视频显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        video_group = QGroupBox("实时视频")
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet('background-color: #1A1826; color: #BAC2DE; font: 16px; border-radius: 6px;')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText('视频准备中...')
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)
        right_layout.addWidget(video_group)
        
        # 设置左右面板的比例
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)
        
        main_widget.setLayout(main_layout)

        # 设置布局边距
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择模型文件', '', '模型文件 (*.pt)')
        if path:
            self.model_path.setText(path)

    def select_source_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, '选择图片目录')
        if dir_path:
            self.source_dir.setText(dir_path)

    def select_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, '选择输出目录')
        if dir_path:
            self.save_dir.setText(dir_path)
            
    def select_video_file(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '', '视频文件 (*.mp4 *.avi *.mov)')
        if path:
            self.video_path.setText(path)
            
    def test_stream_connection(self):
        """测试流媒体连接并显示预览画面"""
        # 先断开现有连接
        if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
            self.disconnect_stream()
            
        stream_url = self.stream_url.text()
        if not stream_url:
            self.log.append('请输入流媒体地址')
            return
            
        self.log.append(f'正在测试连接: {stream_url}')
        
        # 使用OpenCV捕获视频流
        self.preview_cap = cv2.VideoCapture(stream_url)
        if not self.preview_cap.isOpened():
            self.log.append('连接失败，请检查流媒体地址')
            return
            
        self.log.append('连接成功，开始预览...')
        
        # 创建定时器来更新预览画面
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(lambda: self.update_preview(self.preview_cap))
        self.preview_timer.start(30)  # 约30fps
        
        # 显示断开连接按钮
        self.btn_disconnect_stream.setVisible(True)
        self.btn_test_stream.setVisible(False)
        
    def update_preview(self, cap):
        """更新预览画面"""
        ret, frame = cap.read()
        if not ret:
            self.preview_timer.stop()
            cap.release()
            self.log.append('预览结束或连接中断')
            # 重置按钮状态
            self.btn_disconnect_stream.setVisible(False)
            self.btn_test_stream.setVisible(True)
            return
            
        # 转换OpenCV图像为Qt图像并显示
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.KeepAspectRatio))
            
    def disconnect_stream(self):
        """断开流媒体连接"""
        if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
            self.preview_timer.stop()
            if hasattr(self, 'preview_cap') and self.preview_cap.isOpened():
                self.preview_cap.release()
            self.log.append('已断开流媒体连接')
            self.video_label.setText('视频已断开')
            # 重置按钮状态
            self.btn_disconnect_stream.setVisible(False)
            self.btn_test_stream.setVisible(True)
            
    def change_mode(self, mode_text):
        self.mode = {'图片模式': 'image', '视频模式': 'video', '流媒体模式': 'stream'}[mode_text]
        
        # 控制可见性并自动选择路径
        # 图片模式控件
        img_visible = self.mode == 'image'
        self.source_dir.setVisible(img_visible)
        self.btn_source.setVisible(img_visible)
        self.img_label.setVisible(img_visible)
        
        # 视频模式控件
        video_visible = self.mode == 'video'
        self.video_path.setVisible(video_visible)
        self.btn_video.setVisible(video_visible)
        self.video_file_label.setVisible(video_visible)  # 使用正确的变量名控制左侧视频文件标签的可见性
        
        # 流媒体模式控件
        stream_visible = self.mode == 'stream'
        self.stream_url.setVisible(stream_visible)
        self.btn_stream.setVisible(stream_visible)
        self.btn_test_stream.setVisible(stream_visible)
        self.stream_label.setVisible(stream_visible)
        
        # 自动选择路径或设置默认值
        if self.mode == 'image' and not self.source_dir.text():
            self.select_source_dir()
        elif self.mode == 'video' and not self.video_path.text():
            self.select_video_file()
        elif self.mode == 'stream' and self.stream_url.text() == 'rtsp://':
            self.stream_url.setText('rtsp://192.168.1.150/LiveStream/CH0/MainStream')
            
        # 更新状态栏提示
        mode_tips = {
            'image': '图片模式：选择图片目录进行批量检测',
            'video': '视频模式：选择视频文件进行检测',
            'stream': '流媒体模式：输入RTSP等流媒体地址进行实时检测'
        }
        self.statusBar().showMessage(mode_tips[self.mode])
        
        # 重置按钮状态
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(False)
        self.btn_disconnect_stream.setVisible(False)

    def update_conf_label(self, value):
        self.conf_label.setText(f'置信度阈值: {value/100:.2f}')
        
    def update_iou_label(self, value):
        self.iou_label.setText(f'IOU阈值: {value/100:.2f}')
    

    def update_video_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio
        ))
        
    def stop_detection(self):
        """停止当前检测过程"""
        try:
            if self.mode == 'video' and hasattr(self, 'video_detector'):
                # 停止视频检测
                self.log.append('正在停止视频检测...')
                self.video_detector.stop()
                # 恢复按钮状态
                self.btn_start.setVisible(True)
                self.btn_stop.setVisible(False)
            elif self.mode == 'stream' and hasattr(self, 'video_thread'):
                # 停止流媒体检测
                self.log.append('正在停止流媒体检测...')
                self.video_thread.stop()
                self.video_thread.wait(2000)
                # 恢复按钮状态
                self.btn_start.setVisible(True)
                self.btn_stop.setVisible(False)
                self.btn_test_stream.setVisible(True)
                # 重置视频标签
                self.video_label.setText('视频已停止')
        except Exception as e:
            self.log.append(f'停止检测时出错: {str(e)}')

    def start_detection(self):
        try:
            # 停止现有线程
            if hasattr(self, 'video_thread') and self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.wait(2000)

            self.log.append('开始检测...')
            if self.mode == 'image':
                detect_images(
                    model_path=self.model_path.text(),
                    source_dir=self.source_dir.text(),
                    save_dir=self.save_dir.text(),
                    conf=self.conf_slider.value()/100,
                    iou=self.iou_slider.value()/100
                )
                self.log.append('检测完成！结果已保存到：' + self.save_dir.text())
            elif self.mode == 'video':
                # 使用修改后的VideoDetector类
                self.video_detector = detect_video(
                    model_path=self.model_path.text(),
                    video_path=self.video_path.text(),
                    save_dir=self.save_dir.text(),
                    conf=self.conf_slider.value()/100,
                    iou=self.iou_slider.value()/100
                )
                # 连接信号
                self.video_detector.frame_processed.connect(self.update_video_frame)
                self.video_detector.detection_finished.connect(self.log.append)
                
                # 显示停止按钮，隐藏开始按钮
                self.btn_start.setVisible(False)
                self.btn_stop.setVisible(True)
            else:  # 流媒体模式
                from detect_stream import StreamDetector
                # 通过detect_stream创建QThread子类实例
                self.video_thread = detect_stream(
                    model_path=self.model_path.text(),
                    stream_url=self.stream_url.text(),
                    save_dir=self.save_dir.text(),
                    conf=self.conf_slider.value()/100,
                    iou=self.iou_slider.value()/100
                )
                if self.video_thread:
                    self.video_thread.detector.frame_processed.connect(self.update_video_frame)
                    self.video_thread.finished.connect(lambda: self.log.append('检测线程安全退出'))
                    
                    # 显示停止按钮，隐藏开始按钮
                    self.btn_start.setVisible(False)
                    self.btn_stop.setVisible(True)
                    self.btn_disconnect_stream.setVisible(False)
                    self.btn_test_stream.setVisible(False)
        except Exception as e:
            self.log.append(f'错误: {str(e)}')

    def closeEvent(self, event):
        # 窗口关闭时安全终止线程
        if hasattr(self, 'video_thread') and isinstance(self.video_thread, QThread):
            if self.video_thread.isRunning():
                self.video_thread.stop()  # 调用StreamThread的stop方法
                self.video_thread.wait(3000)
        event.accept()
        try:
            # 在布局中添加视频显示区域
            self.layout.addWidget(QLabel('实时视频:'), self.row, 0)
            self.video_label = QLabel()
            self.video_label.setMinimumSize(640, 480)
            self.video_label.setStyleSheet('background-color: #202020; color: #808080; font: 16px')
            self.video_label.setText('视频准备中...')
            self.layout.addWidget(self.video_label, self.row, 1, 1, 2)
            self.row += 1
            
            # 添加视频处理线程
            from detect_stream import StreamDetector
            self.video_thread = StreamDetector()
            self.video_thread = StreamDetector(
                model_path=self.model_path.text(),
                stream_url=self.stream_url.text(),
                save_dir=self.save_dir.text(),
                conf=self.conf_slider.value()/100,
                iou=self.iou_slider.value()/100
            )
            self.video_thread.frame_processed.connect(self.update_video_frame)
            self.log.append('检测完成！结果已保存到：' + self.save_dir.text())
        except Exception as e:
            self.log.append(f'错误: {str(e)}')

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    exe = DetectionApp()
    exe.show()
    sys.exit(app.exec_())


class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def run(self):
        self.timer.start(40)

    def update_frame(self):
        with QMutexLocker(self.detector.queue_lock):
            if self.detector.frame_queue:
                frame = self.detector.frame_queue.popleft()
                self.frame_signal.emit(frame)


# 在UI初始化部分添加
self.video_thread = VideoThread(self.video_thread)
self.video_thread.frame_signal.connect(self.update_video_frame)
QTimer.singleShot(0, self.video_thread.start)
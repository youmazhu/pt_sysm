import sys
import cv2
import os
import time
import shutil
import numpy as np
from collections import deque
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QFileDialog, QWidget, QLabel, QLineEdit, QTextEdit, QComboBox, QSlider, QHBoxLayout, QGridLayout, QVBoxLayout, QGroupBox, QListWidget, QListWidgetItem, QDesktopWidget, QScrollArea, QFrame)
from PyQt5.QtGui import QIcon, QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker, QTimer, QUrl
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QDesktopServices
from detect_images import detect_images
from detect_video import detect_video
from detect_stream import detect_stream
from model_utils import load_model_classes, get_class_names_list

# 定义检测结果管理类
class DetectionResultManager:
    def __init__(self, base_dir="e:\\yolov8_bicycle\\detection_results", max_images_per_class=100):
        self.base_dir = base_dir
        self.max_images_per_class = max_images_per_class
        self.class_dirs = {}
        
    def update_class_dirs(self, class_names):
        """根据模型类别更新目录结构"""
        self.class_dirs = {}
        for class_name in class_names:
            if class_name != "全部":  # 排除"全部"选项
                self.class_dirs[class_name] = os.path.join(self.base_dir, class_name)
                
        # 确保目录存在
        for class_dir in self.class_dirs.values():
            os.makedirs(class_dir, exist_ok=True)
    
    def save_detection_image(self, image, class_name, confidence):
        """保存检测结果图片，并限制每个类别最多保存max_images_per_class张图片"""
        if class_name not in self.class_dirs:
            return None
            
        # 生成文件名：类别_置信度_时间戳.jpg
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{class_name}_{confidence:.2f}_{timestamp}.jpg"
        save_path = os.path.join(self.class_dirs[class_name], filename)
        
        # 保存图片
        cv2.imwrite(save_path, image)
        
        # 检查并删除多余的图片（保留最新的max_images_per_class张）
        self._cleanup_old_images(class_name)
        
        return save_path
    
    def _cleanup_old_images(self, class_name):
        """清理旧图片，只保留最新的max_images_per_class张"""
        if class_name not in self.class_dirs:
            return
            
        class_dir = self.class_dirs[class_name]
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(class_dir, f))]
        
        # 按修改时间排序
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # 删除多余的图片
        if len(files) > self.max_images_per_class:
            for old_file in files[self.max_images_per_class:]:
                try:
                    os.remove(old_file)
                except Exception as e:
                    print(f"删除旧图片失败: {str(e)}")
    
    def get_images_by_class(self, class_name=None):
        """获取指定类别的所有图片路径，如果class_name为None则返回所有图片"""
        result = []
        
        if class_name is not None and class_name != "全部":
            if class_name in self.class_dirs:
                class_dir = self.class_dirs[class_name]
                files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(class_dir, f))]
                # 按修改时间排序（最新的在前）
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                result.extend(files)
        else:
            # 返回所有类别的图片
            for class_name in self.class_dirs:
                result.extend(self.get_images_by_class(class_name))
                
        return result

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_style()
        self.statusBar().showMessage('就绪')  # 初始化状态栏
        
        # 初始化检测结果管理器
        self.result_manager = DetectionResultManager()
        
        # 初始化检测结果存储
        self.detection_results = {}
        
        self.initUI()
        
        # 加载默认模型的类别信息
        default_model_path = self.model_path.text()
        if os.path.exists(default_model_path):
            self.load_model_classes(default_model_path)

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
        self.setGeometry(100, 100, 1400, 900)

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
        
        # 添加检测结果显示区域
        detection_group = QGroupBox("检测结果")
        detection_layout = QHBoxLayout()
        
        # 左侧：检测结果列表
        results_layout = QVBoxLayout()
        
        # 添加类别筛选下拉框
        filter_layout = QHBoxLayout()
        filter_label = QLabel("筛选类别:")
        self.class_filter = QComboBox()
        self.class_filter.addItems(["全部", "motorcycle", "person", "bicycle", "gas_tank"])
        self.class_filter.currentTextChanged.connect(self.filter_detection_results)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.class_filter)
        results_layout.addLayout(filter_layout)
        
        # 添加检测结果列表
        self.results_list = QListWidget()
        self.results_list.setMinimumWidth(300)
        self.results_list.setStyleSheet('background-color: #313244; color: #CDD6F4; border-radius: 6px;')
        self.results_list.itemClicked.connect(self.show_detection_detail)
        results_layout.addWidget(self.results_list)
        
        # 添加统计信息
        self.stats_label = QLabel("检测统计: 0个结果")
        results_layout.addWidget(self.stats_label)
        
        # 右侧：检测结果详情
        detail_layout = QVBoxLayout()
        
        # 添加详情标签
        self.detail_label = QLabel("选择一个检测结果查看详情")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setStyleSheet('background-color: #313244; color: #CDD6F4; padding: 10px; border-radius: 6px;')
        detail_layout.addWidget(self.detail_label)
        
        # 添加图片预览
        self.detail_image = QLabel()
        self.detail_image.setMinimumSize(320, 240)
        self.detail_image.setAlignment(Qt.AlignCenter)
        self.detail_image.setStyleSheet('background-color: #1A1826; color: #BAC2DE; border-radius: 6px;')
        self.detail_image.setText("选择检测结果查看图片")
        detail_layout.addWidget(self.detail_image)
        
        # 添加打开文件夹按钮
        self.btn_open_folder = QPushButton("打开图片所在文件夹")
        self.btn_open_folder.clicked.connect(self.open_result_folder)
        detail_layout.addWidget(self.btn_open_folder)
        
        # 将左右两侧添加到检测结果布局
        detection_layout.addLayout(results_layout, 1)
        detection_layout.addLayout(detail_layout, 2)
        
        detection_group.setLayout(detection_layout)
        right_layout.addWidget(detection_group)
        
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
            # 加载模型类别信息并更新UI
            self.load_model_classes(path)

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
        
    def handle_detection_result(self, image, class_name, confidence):
        """处理检测结果，保存图像并更新UI"""
        try:
            # 保存检测结果图像
            save_path = self.result_manager.save_detection_image(image, class_name, confidence)
            if save_path:
                # 如果当前筛选的类别与检测结果类别相同，或者选择了"全部"，则更新UI
                current_filter = self.class_filter.currentText()
                if current_filter == "全部" or current_filter == class_name:
                    # 创建列表项
                    item = QListWidgetItem(f"{class_name} - 置信度: {confidence:.2f}")
                    item.setData(Qt.UserRole, save_path)  # 存储图片路径
                    self.results_list.insertItem(0, item)  # 在列表顶部插入新项
                    
                    # 更新统计信息
                    self.stats_label.setText(f"检测统计: {self.results_list.count()}个结果")
        except Exception as e:
            self.log.append(f"保存检测结果失败: {str(e)}")
        
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
                    iou=self.iou_slider.value()/100,
                    result_manager=self.result_manager
                )
                # 连接信号
                self.video_detector.frame_processed.connect(self.update_video_frame)
                self.video_detector.detection_finished.connect(self.log.append)
                # 连接检测结果信号
                self.video_detector.detection_result.connect(self.handle_detection_result)
                
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
                    iou=self.iou_slider.value()/100,
                    result_manager=self.result_manager
                )
                if self.video_thread:
                    self.video_thread.detector.frame_processed.connect(self.update_video_frame)
                    self.video_thread.detector.detection_result.connect(self.handle_detection_result)
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
            
    def load_model_classes(self, model_path):
        """加载模型类别信息并更新UI"""
        try:
            # 获取模型类别名称列表
            class_names = get_class_names_list(model_path)
            
            if class_names:
                # 更新类别筛选下拉框
                self.class_filter.clear()
                self.class_filter.addItems(class_names)
                
                # 更新检测结果管理器的类别目录
                self.result_manager.update_class_dirs([name for name in class_names if name != "全部"])
                
                self.log.append(f"已加载模型类别信息: {', '.join([name for name in class_names if name != '全部'])}")
            else:
                self.log.append("未能从模型中读取类别信息，使用默认类别")
                # 使用默认类别
                default_classes = ["全部", "motorcycle", "person", "bicycle", "gas_tank"]
                self.class_filter.clear()
                self.class_filter.addItems(default_classes)
                self.result_manager.update_class_dirs([name for name in default_classes if name != "全部"])
        except Exception as e:
            self.log.append(f"加载模型类别信息失败: {str(e)}")

    def filter_detection_results(self, class_name):
        """根据选择的类别筛选检测结果"""
        try:
            # 获取指定类别的图片
            filtered_images = self.result_manager.get_images_by_class(class_name if class_name != "全部" else None)
            
            # 清空结果列表
            self.results_list.clear()
            
            # 添加筛选后的结果到列表
            for img_path in filtered_images:
                # 从文件名中提取信息
                filename = os.path.basename(img_path)
                parts = filename.split('_')
                if len(parts) >= 2:
                    class_name = parts[0]
                    confidence = float(parts[1]) if len(parts) > 1 else 0.0
                    timestamp = '_'.join(parts[2:]).replace('.jpg', '') if len(parts) > 2 else ''
                    
                    # 创建列表项
                    item = QListWidgetItem(f"{class_name} - 置信度: {confidence:.2f}")
                    item.setData(Qt.UserRole, img_path)  # 存储图片路径
                    self.results_list.addItem(item)
            
            # 更新统计信息
            self.stats_label.setText(f"检测统计: {self.results_list.count()}个结果")
        except Exception as e:
            self.log.append(f"筛选检测结果失败: {str(e)}")

    def show_detection_detail(self, item):
        """显示选中检测结果的详细信息"""
        try:
            # 获取图片路径
            img_path = item.data(Qt.UserRole)
            if not img_path or not os.path.exists(img_path):
                self.detail_label.setText("图片文件不存在")
                return
                
            # 从文件名中提取信息
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            if len(parts) >= 2:
                class_name = parts[0]
                confidence = float(parts[1]) if len(parts) > 1 else 0.0
                timestamp = '_'.join(parts[2:]).replace('.jpg', '') if len(parts) > 2 else ''
                
                # 设置详情标签
                detail_text = f"类别: {class_name}\n置信度: {confidence:.2f}\n时间戳: {timestamp}"
                self.detail_label.setText(detail_text)
                
                # 显示图片
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img.shape
                    bytes_per_line = ch * w
                    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.detail_image.setPixmap(QPixmap.fromImage(q_img).scaled(
                        self.detail_image.width(), self.detail_image.height(), 
                        Qt.KeepAspectRatio))
                else:
                    self.detail_image.setText("无法加载图片")
        except Exception as e:
            self.log.append(f"显示检测详情失败: {str(e)}")

    def open_result_folder(self):
        """打开检测结果所在文件夹"""
        try:
            # 获取当前选中的图片路径
            current_item = self.results_list.currentItem()
            if current_item:
                img_path = current_item.data(Qt.UserRole)
                if img_path and os.path.exists(img_path):
                    # 打开文件所在文件夹
                    folder_path = os.path.dirname(img_path)
                    QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
                    return
                    
            # 如果没有选中项或文件不存在，打开结果根目录
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.result_manager.base_dir))
        except Exception as e:
            self.log.append(f"打开文件夹失败: {str(e)}")


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


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    exe = DetectionApp()
    exe.show()
    sys.exit(app.exec_())
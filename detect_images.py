from ultralytics import YOLO
import os

def detect_images(model_path=r'E:\yolov8_bicycle\0402-3.pt',
                  source_dir=r'E:\yolov8_bicycle\images\w_check',
                  save_dir=r'E:\yolov8_bicycle\runs\detect'):
    """
    对指定目录下的图片进行目标检测
    Args:
        model_path: YOLOv8模型路径，默认使用yolov8n.pt
        source_dir: 待检测图片所在目录
        save_dir: 检测结果保存目录
    """
    try:
        # 加载模型
        model_path=r'E:\yolov8_bicycle\0402-3.pt'
        source_dir=r'E:\yolov8_bicycle\images\w_check'
        save_dir=r'E:\yolov8_bicycle\runs\detect'
        model = YOLO(model_path)
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

        
        print(f'找到 {len(image_files)} 个图片文件，开始处理...')
        
        # 对图片进行批量检测
        results = model(source=source_dir, save=True, save_txt=True, project=save_dir)
        
        print(f'\n检测完成！结果已保存到 {save_dir} 目录')
        print('\n检测结果包含：')
        print('1. 标注后的图片：显示边界框、类别标签和置信度')
        print('2. 检测结果文本文件：包含每个检测对象的详细信息')
        
    except Exception as e:
        print(f'发生错误: {str(e)}')

if __name__ == '__main__':
    # 使用默认参数运行检测
    detect_images()
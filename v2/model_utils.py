from ultralytics import YOLO
import os

def load_model_classes(model_path):
    """
    从YOLOv8模型文件中加载类别信息
    Args:
        model_path: 模型文件路径(.pt文件)
    Returns:
        dict: 类别ID到类别名称的映射字典
    """
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 获取模型的类别名称
        class_names = model.names if hasattr(model, 'names') else {}
        
        return class_names
    except Exception as e:
        print(f"加载模型类别信息失败: {str(e)}")
        return {}

def get_class_names_list(model_path, include_all=True):
    """
    获取模型类别名称列表，用于UI显示
    Args:
        model_path: 模型文件路径
        include_all: 是否包含"全部"选项
    Returns:
        list: 类别名称列表
    """
    class_names = load_model_classes(model_path)
    
    # 将类别ID映射到类别名称的列表
    names_list = list(class_names.values())
    
    # 如果需要，添加"全部"选项
    if include_all and names_list:
        names_list.insert(0, "全部")
        
    return names_list
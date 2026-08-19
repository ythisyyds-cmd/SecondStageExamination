from pathlib import Path
import cv2
from ultralytics import YOLO


#找到项目根目录和两个模型文件
current_file_path = Path(__file__).resolve()           #取得当前文件的完整路径
web_folder_path = current_file_path.parent
project_folder_path = web_folder_path.parent           #当前文件夹的上一层就是项目根目录

detection_model_path = project_folder_path / "yolov8n.pt"
segmentation_model_path = project_folder_path / "yolov8n-seg.pt"

detection_model = YOLO(str(detection_model_path))      #模型放在函数外 网页启动时只加载一次
segmentation_model = YOLO(str(segmentation_model_path))


#把模型结果整理成页面表格需要的形式
def get_result_data(result, task_name):
    result_data = []

    for box in result.boxes:
        class_id = int(box.cls.item())                 #把只有一个元素的张量转换成普通数值作为类别id和置信度
        confidence = box.conf.item()
        class_name = result.names[class_id]

        result_data.append([
            task_name,
            class_name,
            round(confidence, 2)
        ])

    return result_data


#传入图片路径和置信度 返回两种模型处理后的图片和表格数据
def process_image(image_path, confidence_threshold):
    if image_path is None:
        raise ValueError("请先上传一张图片")

    detection_results = detection_model(              #分别用两个模型处理同一张上传图片
        image_path,
        conf=confidence_threshold
    )
    detection_result = detection_results[0]

    segmentation_results = segmentation_model(
        image_path,
        conf=confidence_threshold
    )
    segmentation_result = segmentation_results[0]

    detection_image = detection_result.plot()         #plot得到BGR图片 下一行转成网页显示需要的RGB
    detection_image = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)

    segmentation_image = segmentation_result.plot()
    segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2RGB)

    detection_data = get_result_data(                 #整理两种模型的信息并放进同一个表格
        detection_result,
        "目标检测"
    )
    segmentation_data = get_result_data(
        segmentation_result,
        "图像分割"
    )
    result_data = detection_data + segmentation_data

    return detection_image, segmentation_image, result_data
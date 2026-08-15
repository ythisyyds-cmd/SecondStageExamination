from ultralytics import YOLO

#加载YOLOv8目标检测模型 如果当前文件夹中没有 第一次运行时会自动下载
model = YOLO("yolov8n.pt")

#输入测试图片
pathofimage = "./Object detection and segmentation/test_image.jpg"
results = model(pathofimage)

#目前只有一张测试图片 所以只用取出第一张图片的检测结果
result = results[0]

#显示带有检测框 类别名称和置信度的图片
result.show()

#输出检测到的类别和置信度
for box in result.boxes:                                                #逐个处理检测框
    class_id = int(box.cls.item())                                      #把只有一个元素的张量转换成普通数值作为置信度和id
    confidence = box.conf.item()                                        
    nameofclass = result.names[class_id]
    print(f"检测到：{nameofclass}，置信度：{confidence:.2f}")

#保存带检测框的图片
save_path = "./Object detection and segmentation/outputs/detection_result.jpg"
result.save(filename=save_path)
print("检测结果保存至：", save_path)
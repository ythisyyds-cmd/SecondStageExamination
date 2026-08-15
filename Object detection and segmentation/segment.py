from ultralytics import YOLO


#加载YOLOv8图像分割模型
model = YOLO("yolov8n-seg.pt")


#输入测试图片
pathofimage = "./Object detection and segmentation/test_image.jpg"
results = model(pathofimage)


#目前只有一张测试图片 所以只取第一张图片的分割结果
result = results[0]


#显示带有检测框和分割遮罩的图片
result.show()


#输出分割出的物体类别和置信度
for box in result.boxes:
    class_id = int(box.cls.item())
    confidence = box.conf.item()
    nameofclass = result.names[class_id]

    print(f"分割出的类别：{nameofclass}，置信度：{confidence:.2f}")


#查看模型得到的分割遮罩
if result.masks is not None:
    masks = result.masks.data

    print("分割出的物体数量：", masks.shape[0])
    print("分割遮罩的形状：", masks.shape)
else:
    print("没有检测到可以分割的物体")


#保存带有分割遮罩的图片
save_path = "./Object detection and segmentation/outputs/segmentation_result.jpg"
result.save(filename=save_path)

print("分割结果保存至：", save_path)
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageOps


current_folder = Path(__file__).resolve().parent
parameter_path = current_folder / "mnist_parameters.pth"

parameters = torch.load(                                  #读取训练时保存的四个参数
    parameter_path,
    map_location="cpu",
    weights_only=True
)

w1 = parameters["w1"]
b1 = parameters["b1"]
w2 = parameters["w2"]
b2 = parameters["b2"]


image_transform = transforms.ToTensor()


def sigmoid(x):
    temp = 1 + torch.exp(-x)
    return 1 / temp


def softmax(x):
    max_num = x.max(dim=1, keepdim=True).values
    x = x - max_num
    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim=1, keepdim=True)
    return exp_x / sum_exp


#读取画板内容并预测手写数字
def predict_digit(image_data):
    if image_data is None:
        return "请先写下一个数字", {}

    image = image_data["composite"]                        #取出画板上所有笔画合成后的图片

    if image is None:
        return "请先写下一个数字", {}

    image = image.convert("RGBA")
    white_background = Image.new("RGBA", image.size, "white")
    white_background.alpha_composite(image)                #把透明画布放到白色背景上

    image = white_background.convert("L")
    image = ImageOps.invert(image)                         #转换成MNIST的黑底白字

    content_box = image.getbbox()                          #找出画板中实际写有数字的区域
    if content_box is None:
        return "请先写下一个数字", {}

    image = image.crop(content_box)
    image.thumbnail((20, 20))                              #保持数字原来的比例缩放

    #根据笔迹的像素分布计算数字中心
    pixel_values = torch.tensor(list(image.getdata())).reshape(image.height, image.width)
    total_value = pixel_values.sum()
    x_numbers = torch.arange(image.width)
    y_numbers = torch.arange(image.height)
    center_x = (pixel_values.sum(dim=0) * x_numbers).sum() / total_value
    center_y = (pixel_values.sum(dim=1) * y_numbers).sum() / total_value

    centered_image = Image.new("L", (28, 28), 0)
    left = round(13.5 - center_x.item())
    top = round(13.5 - center_y.item())
    centered_image.paste(image, (left, top))               #把数字放到28×28图片的中央

    image = image_transform(centered_image)

    x = image.reshape(1, 784)                              #把28×28的图片展开成784个像素
    hidden_input = x @ w1 + b1
    hidden_output = sigmoid(hidden_input)

    output_input = hidden_output @ w2 + b2
    probabilities = softmax(output_input)

    prediction = probabilities.argmax(dim=1)
    prediction_num = prediction.item()

    probability_result = {}
    for number in range(10):
        probability = probabilities[0, number].item()
        probability_result[str(number)] = probability

    return f"预测数字：{prediction_num}", probability_result

#清空识别结果
def clear_digit():
    return "预测数字会显示在这里", None

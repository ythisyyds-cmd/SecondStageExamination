import os
import base64                                                    #通过API调用MiMo不支持直传本地文件 只支持Base64
from dotenv import load_dotenv
from openai import OpenAI


#读取.env文件中的API Key
load_dotenv()


#创建连接MiMo的客户端
client = OpenAI(                                                            
    api_key=os.environ.get("MIMO_API_KEY"),
    base_url="https://api.xiaomimimo.com/v1"
)

#把传入图片转换成Base64文本
def image_to_base64(image_path):
    image_file = open(image_path, "rb")                          #用二进制方式打开图片
    image_bytes = image_file.read()                              #读取图片的二进制数据
    image_file.close()                                           #读取完成后关闭图片文件

    base64_bytes = base64.b64encode(image_bytes)                 #把图片数据编码成Base64
    base64_text = base64_bytes.decode("utf-8")                   #把编码结果转换成普通字符串

    return base64_text

#给MiMo发送聊天记录并取出回答
def get_mimo_answer(messages):
    completion = client.chat.completions.create(
        model="mimo-v2.5",
        messages=messages,                                       #把目前为止的所有聊天记录发给模型
        max_completion_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        stream=False,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )

    choices = completion.choices                                 #取出候选回复列表
    first_choice = choices[0]                                    #取出第一个候选回复
    message = first_choice.message                               #取出回复消息
    answer = message.content                                     #取出消息正文

    return answer


#根据图片路径和问题建立图文消息
def create_image_message(image_path, question):
    image_base64 = image_to_base64(image_path)                   #把图片转换成Base64文本

    #根据扩展名确定图片类型
    path_parts = os.path.splitext(image_path)                    #把图片路径拆成文件名和扩展名
    file_extension = path_parts[1]
    file_extension = file_extension.lower()                      #把大写扩展名转换成小写

    if file_extension == ".jpg" or file_extension == ".jpeg":
        image_type = "image/jpeg"
    elif file_extension == ".png":
        image_type = "image/png"
    else:
        raise ValueError("目前只支持JPG、JPEG和PNG格式的图片")

    #给Base64文本添加图片类型和编码信息
    base64_prefix = "data:" + image_type + ";base64,"
    image_data_url = base64_prefix + image_base64


    image_part = {                                               #保存图片内容
        "type": "image_url",
        "image_url": {
            "url": image_data_url
        }
    }

    text_part = {                                                #保存对图片提出的问题
        "type": "text",
        "text": question
    }

    image_user_message = {                                       #把图片和问题合并成一条用户消息
        "role": "user",
        "content": [
            image_part,
            text_part
        ]
    }

    return image_user_message


#设置系统提示 建立聊天记录
system_message = {
    "role": "system",
    "content": "你是小米开发的MiMo智能助手。请使用中文回答。如果无法根据当前对话确定答案，要明确说明不知道，不要随意猜测。"
}
messages = [system_message]                                      #聊天记录一开始只有系统提示



print("输入“图片”上传图片进行问答，输入“清空”清除聊天记录，输入“退出”结束程序")


#不断接收用户输入
while True:
    user_input = input("user：")                                  #等待用户输入消息

    if user_input == "退出":
        print("已结束会话")
        break

    if user_input == "清空":
        messages = [system_message]                              #重新建立只包含系统提示的聊天记录
        print("聊天记录已清空")
        continue                                                 #进入下个循环等待输入

    #根据输入建立文字消息或图文消息
    if user_input == "图片":
        image_path = input("图片路径为：")
        question = input("请针对图片提出问题：")

        user_message = create_image_message(
            image_path=image_path,
            question=question
        )
    else:
        user_message = {
            "role": "user",
            "content": user_input
        }

    messages.append(user_message)                                #把本轮用户消息加入聊天记录
    answer = get_mimo_answer(messages)

    print("MiMo：", answer)
    messages.append({                                            #把模型回答加入聊天记录
        "role": "assistant",
        "content": answer
    })
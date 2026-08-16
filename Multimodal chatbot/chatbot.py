import os

from dotenv import load_dotenv
from openai import OpenAI


#读取.env文件中的API Key
load_dotenv()



client = OpenAI(                                                            
    api_key=os.environ.get("MIMO_API_KEY"),
    base_url="https://api.xiaomimimo.com/v1"
)

#设置系统提示并建立聊天记录
system_message = {
    "role": "system",
    "content": "你是小米开发的MiMo智能助手。请使用中文回答。如果无法根据当前对话确定答案，要明确说明不知道，不要随意猜测。"
}
messages = [system_message]                                      #聊天记录最开始只有系统提示

print("输入“清空”清除聊天记录，输入“退出”结束程序")


#不断接收用户输入
while True:
    user_input = input("user：")                                 #等待用户输入一条消息

    if user_input == "退出":
        print("聊天已结束")
        break

    if user_input == "清空":
        messages = [system_message]                              #重新建立只包含系统提示的聊天记录
        print("聊天记录已清空")
        continue                                                 #跳过后面的代码 重新等待输入

    messages.append({                                            #把本轮用户输入加入聊天记录
        "role": "user",
        "content": user_input
    })

    #调用MiMo模型 这部分参考官方文档
    completion = client.chat.completions.create(
        model="mimo-v2.5",
        messages=messages,                                       #把目前的完整聊天记录发给模型
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

    print("MiMo：", answer)

    messages.append({                                            #把模型回答加入聊天记录
        "role": "assistant",
        "content": answer
    })
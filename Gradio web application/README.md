# Gradio深度学习网页应用

使用Gradio将目标检测与图像分割、多模态聊天机器人和手写数字识别整合到同一个网页中。

目标检测与图像分割页面支持上传图片和调整置信度阈值，使用YOLOv8n和YOLOv8n-seg分别显示检测框、分割遮罩以及类别和置信度。

多模态聊天机器人页面调用mimo-v2.5模型，使用Gradio的Chatbot和MultimodalTextbox组件，支持文字聊天、上下文记忆、图片问答和聊天记录清空。

手写数字识别是自选项目，使用第一阶段手动搭建的BP网络。用户可以直接在画板上写数字，程序会显示预测结果和概率最高的三个数字。

运行网页前需要安装Gradio、Ultralytics、OpenAI等需要的库（详见requirements.txt），并参考.env.example在项目根目录新建.env文件，在其中配置MiMo平台的API Key。准备完成后，在VS Code中直接运行app.py即可启动网页。

第一次运行时会自动下载两个YOLOv8模型参数，启动完成后在浏览器中打开终端显示的本地地址即可。
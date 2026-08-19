import gradio as gr
from vision_functions import process_image                 #从另一个文件导入图片处理函数
from chatbot_functions import (
    create_message_history,
    chat_with_mimo,
    clear_chat
)

#搭建目标检测和图像分割页面
with gr.Blocks(title="Second Stage Examination") as demo:
    gr.Markdown("# app for my code")

    with gr.Tab("目标检测与图像分割"):
        gr.Markdown("上传一张图片，模型会分别完成目标检测和图像分割。")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="filepath",                       #把上传图片的临时路径传给处理函数
                    sources=["upload"],
                    label="上传图片"
                )


            with gr.Column():
                with gr.Row():
                    detection_output = gr.Image(
                        label="目标检测结果"
                    )

                    segmentation_output = gr.Image(
                        label="图像分割结果"
                    )

        with gr.Row():
            confidence_threshold = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="置信度阈值",
                scale=4
            )

            process_button = gr.Button(
                value="开始处理",
                variant="primary",
                scale=1,
                min_width=180
            )

        result_table = gr.Dataframe(
            headers=["处理类型", "类别", "置信度"],
            datatype=["str", "str", "number"],
            interactive=False,
            label="识别结果"
        )

        clear_button = gr.ClearButton(
            components=[
                input_image,
                detection_output,
                segmentation_output,
                result_table
            ],
            value="清空"
        )

        process_button.click(                              #点击后把两个输入交给process_image
            fn=process_image,
            inputs=[
                input_image,
                confidence_threshold
            ],
            outputs=[
                detection_output,
                segmentation_output,
                result_table
            ]
        )


#搭建多模态聊天机器人页面
    with gr.Tab("多模态聊天机器人"):
        gr.Markdown("可以进行文字聊天，也可以上传一张图片并针对图片提问。")

        api_messages = gr.State(
            value=create_message_history                   #为每个网页会话分别建立API聊天记录
        )

        chatbot = gr.Chatbot(
            value=[],
            label="聊天记录",
            height=500,
            layout="bubble",
            placeholder="请输入消息开始聊天"
        )

        with gr.Accordion("查看本轮问题分析", open=False):
            analysis_output = gr.Markdown()

        message_input = gr.MultimodalTextbox(
            sources=["upload"],
            file_types=["image"],
            file_count="single",
            lines=2,
            max_lines=6,
            placeholder="输入消息或上传图片(JPG、JPEG或PNG格式)",
            label="输入消息",
            submit_btn="发送"
        )

        clear_chat_button = gr.Button(
            value="清空聊天记录"
        )

        message_input.submit(                               #按下回车或点击发送后调用聊天函数
            fn=chat_with_mimo,
            inputs=[
                message_input,
                chatbot,
                api_messages
            ],
            outputs=[
                chatbot,
                api_messages,
                analysis_output,
                message_input
            ]
        )

        clear_chat_button.click(                            #同时清空页面内容和API聊天记录
            fn=clear_chat,
            outputs=[
                chatbot,
                api_messages,
                analysis_output,
                message_input
            ]
        )

if __name__ == "__main__":
    demo.launch()                                           #只有直接运行app.py时才启动网页

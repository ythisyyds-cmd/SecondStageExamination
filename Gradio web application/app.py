import gradio as gr
from vision_functions import process_image                 #从另一个文件导入图片处理函数


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


if __name__ == "__main__":
    demo.launch()                                         #只有直接运行app.py时才启动网页

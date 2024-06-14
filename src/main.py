from components import *
from prompting import prompt_output
import gradio
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    with gradio.Blocks() as app:

        title_row()

        prompt_options: tuple[gradio.Checkbox, gradio.Dropdown] = prompt_options_row()

        rag_checkbox, task_selector = prompt_options

        input_text: gradio.Textbox = input_prompt_row()

        submit_button: gradio.Button = submit_row()

        output_text: gradio.Textbox = chatbot_response_row()
    
        input_text.change(fn=input_changed, inputs=[input_text, task_selector], outputs=submit_button)

        task_selector.change(fn=input_changed, inputs=[input_text, task_selector], outputs=submit_button)

        submit_button.click(fn=prompt_output, inputs=[rag_checkbox, task_selector, input_text], outputs=output_text)

    app.launch(share=True)
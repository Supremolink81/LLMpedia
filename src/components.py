import gradio

def title_row() -> None:

    """
    Renders the title row.
    """

    with gradio.Row():

        with gradio.Column():

            pass

        with gradio.Column():

            _: gradio.TextArea = gradio.TextArea("LLMpedia", lines=1, max_lines=1)

        with gradio.Column():

            pass

def prompt_options_row() -> tuple[gradio.Checkbox, gradio.Dropdown]:

    """
    Renders the row for prompt options and returns its components.
    """

    with gradio.Row():

        with gradio.Column():

            rag_checkbox: gradio.Checkbox = gradio.Checkbox(label="Enable RAG?", scale=1)

        with gradio.Column():

            task_selector: gradio.Dropdown = gradio.Dropdown(choices=[
                "Question Answering",
                "Summarization",
               "Reasoning"
            ], label="Task Type")
    
    return rag_checkbox, task_selector

def input_prompt_row() -> gradio.Textbox:

    """
    Renders the row for input and returns its component.
    """

    with gradio.Row():

        input_text: gradio.Textbox = gradio.Textbox(placeholder="Enter prompt here...")

    return input_text

def submit_row() -> gradio.Button:

    """
    Renders the row for the submit button and returns its component.
    """

    with gradio.Row():

        with gradio.Column():
                
            pass

        with gradio.Column():

            submit_button: gradio.Button = gradio.Button("Submit", interactive=False)

        with gradio.Column():

            pass
    
    return submit_button

def chatbot_response_row() -> gradio.Textbox:

    """
    Renders the row for the chatbot response and returns its component.
    """

    with gradio.Row():

        chatbot_response: gradio.Textbox = gradio.Textbox(placeholder="Chatbot output here...", interactive=False)
    
    return chatbot_response

def input_changed(prompt_text: str, selected_task: str) -> None:

    TASK_SELECTED: bool = selected_task != ""

    TEXT_ENTERED: bool = prompt_text != ""

    return gradio.Button("Submit", interactive=TASK_SELECTED and TEXT_ENTERED)
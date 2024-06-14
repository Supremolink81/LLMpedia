import gradio
from typing import Literal
from pinecone_utils import query_pinecone_db
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
import os

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

langchain_agent: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

def get_chatbot_response(prompt: str) -> str:

    """
    Returns the chatbot's response to a prompt.

    Parameters:

        `str` prompt: the prompt to give to the chatbot.

    Returns:

        The chatbot's response as a `str`.
    """

    chatbot_response_message: BaseMessage = langchain_agent.invoke(prompt)

    return chatbot_response_message.content

def prompt_output(use_rag: bool, selected_task: Literal["Question Answering","Summarization","Reasoning"], input_text: str) -> gradio.Textbox:

    """
    Returns a chatbot response to a prompt and associated settings.

    Parameters:

        `bool` use_rag: whether to use Retrieval Augmented Generation.

        `Literal["Question Answering","Summarization","Reasoning"]` selected_task: the task to perform with the prompt.

        `str` input_text: the text passed as input to the chatbot.

    Returns:

        A Gradio `Textbox` containing the chatbot's response.
    """

    chatbot_response: str

    if selected_task == "Question Answering":

        chatbot_response = question_answering_output(use_rag, input_text)

    elif selected_task == "Summarization":

        chatbot_response = summarization_output(input_text)

    else:

        chatbot_response = reasoning_output(use_rag, input_text)

    return gradio.Textbox(chatbot_response, interactive=False)

def question_answering_output(use_rag: bool, input_text: str) -> str:

    """
    Answers the given question, optionally using Retrieval Augmented Generation.

    Parameters:

        `bool` use_rag: whether to use Retrieval Augmented Generation.

        `str` input_text: the text passed as input to the chatbot.

    Returns:

        A `str` containing the answer.
    """

    rag_results: list[str] = []

    if use_rag:

        rag_results = query_pinecone_db(input_text)

    prompt_first_half: str = 'Answer the following question with the given context as a guide.\n\nQuestion: '

    prompt_second_half: str = input_text + "\n\nContext: " + '\n'.join(rag_results)

    prompt: str = prompt_first_half + prompt_second_half

    return get_chatbot_response(prompt)

def summarization_output(input_text: str) -> gradio.Textbox:

    """
    Summarizes the given text.

    Parameters:

        `bool` use_rag: whether to use Retrieval Augmented Generation.

        `str` input_text: the text passed as input to the chatbot.

    Returns:

        A `str` with the summary.
    """

    prompt: str = f"Summarize the given text: \n {input_text}"

    return get_chatbot_response(prompt)

def reasoning_output(use_rag: bool, input_text: str) -> gradio.Textbox:

    """
    Reasons about a particular problem.

    Parameters:

        `bool` use_rag: whether to use Retrieval Augmented Generation.

        `str` input_text: the text passed as input to the chatbot.

    Returns:

        A `str` with the reasoned response.
    """

    rag_results: list[str] = []

    if use_rag:

        rag_results = query_pinecone_db(input_text)

    prompt_first_half: str = 'Reason about the following problem with the context as a guide. Think step by step through the solution.\n\nProblem: '

    prompt_second_half: str = input_text + "\n\nContext: " + '\n'.join(rag_results)

    prompt: str = prompt_first_half + prompt_second_half

    return get_chatbot_response(prompt)
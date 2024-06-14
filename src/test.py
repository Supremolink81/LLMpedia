import unittest
from unittest import mock
import pinecone
import openai.resources
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from prompting import prompt_output

class TestAppLogicNoRAGQuestionAnswering(unittest.TestCase):

    @mock.patch.object(openai.resources.Embeddings, "create")
    @mock.patch.object(ChatOpenAI, "invoke")
    @mock.patch.object(pinecone.Index, "query")
    def test(self, pinecone_query_mock: mock.Mock, chat_invoke_mock: mock.Mock, embeddings_create_mock: mock.Mock) -> None:

        pinecone_query_mock.return_value = {
            "matches": [
                {
                    "id": "BigID",
                    "metadata": {
                        "content": "Epic Blue Sky"
                    }
                }
            ]
        }

        chat_invoke_mock.return_value = AIMessage(content="The sky is blue.")

        embeddings_create_mock.return_value = mock.Mock()

        embeddings_create_mock.return_value.data = [mock.Mock()]

        embeddings_create_mock.return_value.data[0].embedding = [1, 2, 3, 4]

        prompt_output(False, "Question Answering", "What color is the sky?")

        mock_prompt: str = f"Answer the following question with the given context as a guide.\n\nQuestion: What color is the sky?\n\nContext: "

        pinecone_query_mock.assert_not_called()

        embeddings_create_mock.assert_not_called()

        chat_invoke_mock.assert_called_once_with(mock_prompt)

class TestAppLogicRAGQuestionAnswering(unittest.TestCase):

    @mock.patch.object(openai.resources.Embeddings, "create")
    @mock.patch.object(ChatOpenAI, "invoke")
    @mock.patch.object(pinecone.Index, "query")
    def test(self, pinecone_query_mock: mock.Mock, chat_invoke_mock: mock.Mock, embeddings_create_mock: mock.Mock) -> None:

        pinecone_query_mock.return_value = {
            "matches": [
                {
                    "id": "BigID",
                    "metadata": {
                        "content": "Epic Blue Sky"
                    }
                }
            ]
        }

        chat_invoke_mock.return_value = AIMessage(content="The sky is blue.")

        embeddings_create_mock.return_value = mock.Mock()

        embeddings_create_mock.return_value.data = [mock.Mock()]

        embeddings_create_mock.return_value.data[0].embedding = [1, 2, 3, 4]

        prompt_output(True, "Question Answering", "What color is the sky?")

        mock_prompt: str = f"Answer the following question with the given context as a guide.\n\nQuestion: What color is the sky?\n\nContext: Epic Blue Sky"
        
        pinecone_query_mock.assert_called_once_with(
            vector=[1, 2, 3, 4],
            top_k=5,
            include_metadata=True,
            include_values=False
        )

        chat_invoke_mock.assert_called_once_with(mock_prompt)

        embeddings_create_mock.assert_called_once_with(input="What color is the sky?", model="text-embedding-3-small")

class TestAppLogicNoRAGSummarization(unittest.TestCase):

    @mock.patch.object(openai.resources.Embeddings, "create")
    @mock.patch.object(ChatOpenAI, "invoke")
    @mock.patch.object(pinecone.Index, "query")
    def test(self, pinecone_query_mock: mock.Mock, chat_invoke_mock: mock.Mock, embeddings_create_mock: mock.Mock) -> None:

        pinecone_query_mock.return_value = {
            "matches": [
                {
                    "id": "BigID",
                    "metadata": {
                        "content": "Epic Blue Sky"
                    }
                }
            ]
        }

        chat_invoke_mock.return_value = AIMessage(content="The sky is blue.")

        embeddings_create_mock.return_value = mock.Mock()

        embeddings_create_mock.return_value.data = [mock.Mock()]

        embeddings_create_mock.return_value.data[0].embedding = [1, 2, 3, 4]

        prompt_output(False, "Summarization", "What color is the sky?")

        mock_prompt: str = f"Summarize the given text: \n What color is the sky?"

        pinecone_query_mock.assert_not_called()

        embeddings_create_mock.assert_not_called()

        chat_invoke_mock.assert_called_once_with(mock_prompt)

class TestAppLogicRAGSummarization(unittest.TestCase):

    @mock.patch.object(openai.resources.Embeddings, "create")
    @mock.patch.object(ChatOpenAI, "invoke")
    @mock.patch.object(pinecone.Index, "query")
    def test(self, pinecone_query_mock: mock.Mock, chat_invoke_mock: mock.Mock, embeddings_create_mock: mock.Mock) -> None:

        pinecone_query_mock.return_value = {
            "matches": [
                {
                    "id": "BigID",
                    "metadata": {
                        "content": "Epic Blue Sky"
                    }
                }
            ]
        }

        chat_invoke_mock.return_value = AIMessage(content="The sky is blue.")

        embeddings_create_mock.return_value = mock.Mock()

        embeddings_create_mock.return_value.data = [mock.Mock()]

        embeddings_create_mock.return_value.data[0].embedding = [1, 2, 3, 4]

        prompt_output(True, "Summarization", "What color is the sky?")

        mock_prompt: str = f"Summarize the given text: \n What color is the sky?"

        embeddings_create_mock.assert_not_called()

        pinecone_query_mock.assert_not_called()

        chat_invoke_mock.assert_called_once_with(mock_prompt)

class TestAppLogicNoRAGReasoning(unittest.TestCase):

    @mock.patch.object(openai.resources.Embeddings, "create")
    @mock.patch.object(ChatOpenAI, "invoke")
    @mock.patch.object(pinecone.Index, "query")
    def test(self, pinecone_query_mock: mock.Mock, chat_invoke_mock: mock.Mock, embeddings_create_mock: mock.Mock) -> None:

        pinecone_query_mock.return_value = {
            "matches": [
                {
                    "id": "BigID",
                    "metadata": {
                        "content": "Epic Blue Sky"
                    }
                }
            ]
        }

        chat_invoke_mock.return_value = AIMessage(content="The sky is blue.")

        embeddings_create_mock.return_value = mock.Mock()

        embeddings_create_mock.return_value.data = [mock.Mock()]

        embeddings_create_mock.return_value.data[0].embedding = [1, 2, 3, 4]

        prompt_output(False, "Reasoning", "What color is the sky?")

        mock_prompt: str = f"Reason about the following problem with the context as a guide. Think step by step through the solution.\n\nProblem: What color is the sky?\n\nContext: " 

        embeddings_create_mock.assert_not_called()

        pinecone_query_mock.assert_not_called()

        chat_invoke_mock.assert_called_once_with(mock_prompt)

class TestAppLogicRAGReasoning(unittest.TestCase):

    @mock.patch.object(openai.resources.Embeddings, "create")
    @mock.patch.object(ChatOpenAI, "invoke")
    @mock.patch.object(pinecone.Index, "query")
    def test(self, pinecone_query_mock: mock.Mock, chat_invoke_mock: mock.Mock, embeddings_create_mock: mock.Mock) -> None:

        pinecone_query_mock.return_value = {
            "matches": [
                {
                    "id": "BigID",
                    "metadata": {
                        "content": "Epic Blue Sky"
                    }
                }
            ]
        }

        chat_invoke_mock.return_value = AIMessage(content="The sky is blue.")

        embeddings_create_mock.return_value = mock.Mock()

        embeddings_create_mock.return_value.data = [mock.Mock()]

        embeddings_create_mock.return_value.data[0].embedding = [1, 2, 3, 4]

        prompt_output(True, "Reasoning", "What color is the sky?")

        mock_prompt: str = f"Reason about the following problem with the context as a guide. Think step by step through the solution.\n\nProblem: What color is the sky?\n\nContext: Epic Blue Sky" 
        
        pinecone_query_mock.assert_called_once_with(
            vector=[1, 2, 3, 4],
            top_k=5,
            include_metadata=True,
            include_values=False
        )

        embeddings_create_mock.assert_called_once_with(input="What color is the sky?", model="text-embedding-3-small")

        chat_invoke_mock.assert_called_once_with(mock_prompt)

if __name__ == "__main__":

    unittest.main()
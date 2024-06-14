import openai.resources
import pinecone
import datasets
import os
import openai
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_ID: str = os.environ["PINECONE_INDEX_ID"]

def initialize_pinecone_db() -> None:

    """
    Initializes the Pinecone vector database.
    """

    dataset_link: str = "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K"

    # only split category in dataset is train
    embedding_dataset: datasets.Dataset = datasets.load_dataset(path=dataset_link, streaming=True)["train"]

    pinecone_client = pinecone.Pinecone(PINECONE_API_KEY)

    pinecone_index: pinecone.Index = pinecone_client.Index(PINECONE_INDEX_ID)

    for index, entry in enumerate(embedding_dataset):

        entry_embedding_vector: list = entry["openai"]

        entry_text: str = entry["text"]
        
        entry_id: str = entry["_id"]

        print(f"Entry {index+1} with ID {entry_id}")

        pinecone_entry: dict = {
            "id": entry_id.encode("utf-8", "ignore").decode("ascii", "ignore"),
            "values": entry_embedding_vector,
            "metadata": {
                "content": entry_text
            }
        }

        pinecone_index.upsert([pinecone_entry])

def query_pinecone_db(prompt: str) -> list[str]:

    """
    Queries the Pinecone vector database for documents to use for RAG.

    Parameters:

        `str` prompt: the prompt to use for querying.

    Returns:

        A list of documents.
    """

    OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

    openai_client: openai.OpenAI = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt_embedding: list[float] = openai_client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    ).data[0].embedding

    pinecone_client = pinecone.Pinecone(PINECONE_API_KEY)

    pinecone_index: pinecone.Index = pinecone_client.Index(PINECONE_INDEX_ID)

    pinecone_query_response: pinecone.QueryResponse = pinecone_index.query(
        vector=prompt_embedding,
        top_k=5,
        include_metadata=True,
        include_values=False
    )

    return [matched_document["metadata"]["content"] for matched_document in pinecone_query_response["matches"]]

if __name__ == "__main__":

    initialize_pinecone_db()
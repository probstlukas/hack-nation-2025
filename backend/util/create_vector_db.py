import os
import openai
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from util.pdf_text_extraction import parse_pdf_with_nice_tables


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)


def get_pdf_name(pdf_path: str) -> str:
    """Return the PDF file name without extension."""
    return os.path.splitext(os.path.basename(pdf_path))[0]


def create_chroma_vector_db(
    pdf_path: str,
    persist_directory: str = "chroma_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    openai_api_key: str = None,
):
    """
    Creates a Chroma vector database from the given (potentially huge) text using OpenAI's text-embedding-3-small API.
    The text is automatically split into manageable chunks.

    Args:
        text (str): The input text to embed and store.
        collection_name (str): Name of the Chroma collection.
        persist_directory (str): Directory to persist the Chroma DB.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.
        openai_api_key (str): OpenAI API key (if not set in environment).

    Returns:
        chromadb.api.models.Collection.Collection: The created Chroma collection.
    """
    collection_name = get_pdf_name(pdf_path)

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=persist_directory)

    existing_collections = [col.name for col in client.list_collections()]
    if collection_name in existing_collections:
        print(f"Search for existing vectordb '{collection_name}'")
        collection = client.get_collection(collection_name)
        return collection
    else:
        print(f"Create new vectordb with name '{collection_name}'")
        collection = client.create_collection(collection_name)

    if openai_api_key:
        openai.api_key = openai_api_key
    else:
        openai.api_key = OPENAI_API_KEY

    print("Parse pdf...")
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    text = parse_pdf_with_nice_tables(pdf_path)
    chunks = splitter.split_text(text)
    print(f"parsed pdf into {len(chunks)} chunks")

    # Get embeddings for all chunks
    print("Create embeddings...")
    embeddings = []
    batch_size = 100  # OpenAI API supports batching
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]
        response = openai.embeddings.create(input=batch, model="text-embedding-3-small")
        embeddings.extend([item.embedding for item in response.data])

    # Add documents and embeddings
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"Created vectordb with name '{collection_name}'")
    # Persist DB

    return collection

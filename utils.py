from pypdf import PdfReader
from openai import OpenAI
import numpy as np

def load_pdf(file_path: str) -> str:
    """Load a PDF file and return its text content as a string."""

    reader = PdfReader(file_path)
    text = ''

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
        except Exception as e:
            print(f"Error extracting text from page {i}: {e}")

    return text

def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split the input text into chunks of specified size with overlap."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


client = OpenAI()

def get_embedding(text: str, ) -> list[float]:
    """Get the embedding vector for the given text using OpenAI's API."""

    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_index(chunks):
    index = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        index.append({
            "text": chunk,
            "embedding": emb
        })
    return index


def retrieve(query, index, top_k=3):
    query_emb = get_embedding(query)

    scores = []
    for item in index:
        score = cosine_similarity(query_emb, item["embedding"])
        scores.append((score, item["text"]))

    scores.sort(reverse=True, key=lambda x: x[0])

    return [text for score, text in scores[:top_k]]
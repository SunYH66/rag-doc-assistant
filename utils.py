from pypdf import PdfReader
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

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


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt

def generate_answer(prompt: str) -> str:
    """Generate an answer using the LLM based on the prompt."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def rag_pipeline(query: str, index, top_k: int = 3) -> str:
    """Run the full RAG pipeline: retrieve -> build prompt -> generate answer."""

    retrieved_chunks = retrieve(query, index, top_k=top_k)
    prompt = build_prompt(query, retrieved_chunks)
    answer = generate_answer(prompt)

    return answer
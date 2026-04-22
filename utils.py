from pypdf import PdfReader

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
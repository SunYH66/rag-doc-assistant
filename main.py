from pathlib import Path
from utils import load_pdf, split_text_into_chunks

def main():
    file_path = Path('file') / 'diffusion_model.pdf'
    text = load_pdf(file_path)
    chunks = split_text_into_chunks(text)

    print(f'Total text length: {len(text)} characters')
    print(f'Number of chunks: {len(chunks)}')
    print(f'First chunk:\n{chunks[0]}')

if __name__ == '__main__':
    main()
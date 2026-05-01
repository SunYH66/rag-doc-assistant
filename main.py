from pathlib import Path
from utils import load_pdf, split_text_into_chunks, build_index, rag_pipeline


if __name__ == '__main__':
    pdf_path = Path('file') / 'diffusion_model.pdf'  # 换成你的PDF路径

    text = load_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    index = build_index(chunks)

    while True:
        query = input("\nAsk a question: ")

        if query.lower() in ["exit", "quit"]:
            break

        answer = rag_pipeline(query, index, top_k=3)

        print("\nAnswer:")
        print(answer)
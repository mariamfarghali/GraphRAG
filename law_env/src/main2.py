import asyncio
import os
from src.trial2 import HFTokenizerAdapter, TextChunker, FileHandler, SentenceTransformerEmbedder


async def main():
    # ------------------------
    # Initialize components
    # ------------------------
    tokenizer = HFTokenizerAdapter(model_name="aubmindlab/bert-base-arabertv2")
    chunker = TextChunker(tokenizer, max_tokens=400, min_tokens=50)
    file_handler = FileHandler()
    embedder = SentenceTransformerEmbedder("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # ------------------------
    # Read input document
    # ------------------------
    filename = r"C:\LDC\GRAG_newLaborLaw\law_env\raw_data\newLow.txt"
    content = await file_handler.read_file(filename)

    # ------------------------
    # Chunk the content
    # ------------------------
    chunks = await chunker.chunk_text(content)

    # ------------------------
    # Save chunks with token size
    # ------------------------
    output_path = os.path.join(os.path.dirname(filename), "chunked_output2.txt")
    await file_handler.write_chunks_to_file(chunks, output_path, tokenizer)

    # ------------------------
    # Generate embeddings
    # ------------------------
    print("Generating embeddings...")
    embeddings = await embedder.embed_texts(chunks)

    # ------------------------
    # Final Stats
    # ------------------------
    print(f"\n--- Processing Summary ---")
    print(f"Original token count: {await tokenizer.count_tokens(content)}")
    print(f"Number of chunks created: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Chunks saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

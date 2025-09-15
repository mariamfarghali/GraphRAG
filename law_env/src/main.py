from src.trial import read_file, chunk_text, write_chunks_to_file, count_tokens
import os

def test_chunking():
    # Use raw string or forward slashes for Windows paths
    filename = r"C:\LDC\GRAG_newLaborLaw\law_env\raw_data\newLow.txt"

    # Alternative: Use forward slashes
    # filename = "C:/LDC/GRAG_newLaborLaw/law_env/raw_data/newLow.txt"

    # Read the Arabic document
    content = read_file(filename)

    # Chunk the content
    chunks = chunk_text(content)

    # Write chunks to a file for inspection
    output_path = os.path.join(os.path.dirname(filename), "chunked_output.txt")
    write_chunks_to_file(chunks, output_path)

    # Print summary
    print(f"Original document token count: {count_tokens(content)}")
    print(f"Number of chunks created: {len(chunks)}")

    # Display token counts for each chunk
    for i, chunk in enumerate(chunks):
        token_count = count_tokens(chunk)
        print(f"Chunk {i+1}: {token_count} tokens")

        # Verify chunk sizes meet requirements
        if token_count > 500:
            print(f"  ERROR: Chunk {i+1} exceeds 500 tokens!")
        elif token_count <= 50:
            print(f"  WARNING: Chunk {i+1} has <=50 tokens!")

if __name__ == "__main__":
    test_chunking()
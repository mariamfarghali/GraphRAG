import re
import os
from typing import List, Union
import torch
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

class TextSplitter:
    """Responsible for splitting text into paragraphs and word-boundary pieces."""

    def __init__(self, tokenizer_name: str = "xlm-roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if text else 0

    def split_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    def split_text_at_word_boundary(self, text: str, max_tokens: int) -> List[str]:
        words = text.split()
        chunks, current, current_tokens = [], [], 0

        for w in words:
            w_tokens = self.count_tokens(w + " ")
            if current_tokens + w_tokens > max_tokens:
                if current:
                    chunks.append(" ".join(current))
                current = [w]
                current_tokens = w_tokens
            else:
                current.append(w)
                current_tokens += w_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

class TextChunker:
    """Responsible for applying chunking rules."""

    def __init__(self, splitter: TextSplitter, max_tokens: int = 400, min_tokens: int = 50):
        self.splitter = splitter
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    async def chunk_text(self, text: str) -> List[str]:
        paragraphs = self.splitter.split_paragraphs(text)
        chunks, buffer = [], ""

        def buffer_tokens():
            return self.splitter.count_tokens(buffer) if buffer else 0

        # Iterate paragraphs
        for paragraph in paragraphs:
            p_toks = self.splitter.count_tokens(paragraph)

            # CASE A: paragraph too large
            if p_toks > self.max_tokens:
                pieces = self.splitter.split_text_at_word_boundary(paragraph, self.max_tokens)
                for piece in pieces:
                    pt = self.splitter.count_tokens(piece)
                    if pt <= self.min_tokens:
                        buffer = (buffer + " " + piece).strip() if buffer else piece
                    else:
                        if buffer and buffer_tokens() + pt <= self.max_tokens:
                            chunks.append((buffer + " " + piece).strip())
                            buffer = ""
                        else:
                            if buffer:
                                merged = (buffer + " " + piece).strip()
                                chunks.extend(self.splitter.split_text_at_word_boundary(merged, self.max_tokens))
                                buffer = ""
                            else:
                                chunks.append(piece)

            # CASE B: too small
            elif p_toks <= self.min_tokens:
                buffer = (buffer + " " + paragraph).strip() if buffer else paragraph

            # CASE C: normal paragraph
            else:
                if buffer and buffer_tokens() + p_toks <= self.max_tokens:
                    chunks.append((buffer + " " + paragraph).strip())
                    buffer = ""
                elif buffer:
                    merged = (buffer + " " + paragraph).strip()
                    chunks.extend(self.splitter.split_text_at_word_boundary(merged, self.max_tokens))
                    buffer = ""
                else:
                    chunks.append(paragraph)

        # Finalize buffer
        if buffer:
            if chunks and self.splitter.count_tokens(chunks[-1]) + buffer_tokens() <= self.max_tokens:
                chunks[-1] = chunks[-1] + " " + buffer
            else:
                chunks.extend(self.splitter.split_text_at_word_boundary(buffer, self.max_tokens))

        # Ensure no chunk < min_tokens
        i = 0
        while i < len(chunks):
            t = self.splitter.count_tokens(chunks[i])
            if t <= self.min_tokens:
                if i > 0 and self.splitter.count_tokens(chunks[i - 1]) + t <= self.max_tokens:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    del chunks[i]
                    continue
                elif i + 1 < len(chunks):
                    chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                    del chunks[i]
                    continue
            i += 1

        return chunks

class Embedder:
    def __init__(
        self,
        embedder_model: Union[str, SentenceTransformer] = "LLukas22/paraphrase-multilingual-mpnet-base-v2-embedding-all",
        batch_size: int = 32,
        to_numpy: bool = True,
    ):
        """
        Embedder class for encoding text chunks.
        Args:
            embedder_model (Union[str, SentenceTransformer]): Hugging Face model name or preloaded model.
            batch_size (int): Batch size for embeddings.
            to_numpy (bool): Return numpy arrays if True, else torch tensors.
        """
        self.batch_size = batch_size
        self.to_numpy = to_numpy

        # Allow injection of preloaded SentenceTransformer model OR model name
        if isinstance(embedder_model, str):
            self.model = SentenceTransformer(embedder_model)
            self.model_name = embedder_model
        elif isinstance(embedder_model, SentenceTransformer):
            self.model = embedder_model
            self.model_name = embedder_model.__class__.__name__
        else:
            raise ValueError("embedder_model must be a str or SentenceTransformer instance")

    async def embed(self, chunks: List[str]) -> Union[np.ndarray, torch.Tensor]:
        """
        Embed a list of text chunks.
        Args:
            chunks (List[str]): List of text chunks.
        Returns:
            Union[np.ndarray, torch.Tensor]: Embeddings of shape (num_chunks, embedding_dim).
        """
        if not chunks:
            return np.array([]) if self.to_numpy else torch.empty(0)

        embeddings = self.model.encode(
            chunks,
            batch_size=self.batch_size,
            convert_to_tensor=not self.to_numpy,
            show_progress_bar=True,
        )

        print(f"Embedded {len(chunks)} chunks. Shape: {embeddings.shape}")
        return embeddings

def read_file(filename: str) -> str:
    with open(os.path.normpath(filename), 'r', encoding='utf-8') as f:
        return f.read()

def write_chunks_to_file(chunks: List[str], output_filename: str, splitter: TextSplitter):
    with open(os.path.normpath(output_filename), 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"=== Chunk {i+1} === (Tokens: {splitter.count_tokens(chunk)})\n")
            f.write(chunk + "\n\n")

import re
from typing import List
import asyncio
from abc import ABC, abstractmethod
from transformers import AutoTokenizer


# ------------------------
# Tokenizer Adapter
# ------------------------
class Tokenizer(ABC):
    """Abstract base class for tokenizers"""

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    async def split_text(self, text: str) -> List[str]:
        pass


# ------------------------
# HuggingFace Tokenizer
# ------------------------
class HFTokenizer(Tokenizer):
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    async def split_text(self, text: str) -> List[str]:
        """Split text hierarchically into chunks"""
        paragraphs = re.split(r"\n\s*\n", text.strip())  # Paragraphs
        chunks = []

        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())  # Sentences
            buffer = ""

            for sentence in sentences:
                words = sentence.split()
                for word in words:
                    # Build buffer progressively
                    candidate = (buffer + " " + word).strip()
                    tokens = await self.count_tokens(candidate)

                    if tokens > 400:  # too long → split
                        if buffer:
                            chunks.extend(await self._finalize_chunk(buffer))
                        buffer = word  # start new with current word
                    else:
                        buffer = candidate

                # End of sentence
                if buffer:
                    chunks.extend(await self._finalize_chunk(buffer))
                    buffer = ""

            if buffer:  # leftover after paragraph
                chunks.extend(await self._finalize_chunk(buffer))

        # Merge small chunks (<50 tokens)
        merged_chunks = await self._merge_small_chunks(chunks)
        return merged_chunks

    async def _finalize_chunk(self, text: str) -> List[str]:
        """Apply rules on a candidate chunk"""
        tokens = await self.count_tokens(text)

        if tokens > 400:
            return await self._split_long_chunk(text)
        elif tokens <= 50:
            return [("__SMALL__", text)]  # mark for merging later
        else:
            return [text]

    async def _split_long_chunk(self, text: str) -> List[str]:
        """Split recursively into ≤ 400 chunks"""
        words = text.split()
        current, result = "", []

        for word in words:
            candidate = (current + " " + word).strip()
            tokens = await self.count_tokens(candidate)

            if tokens > 400:
                if current:
                    result.append(current)
                current = word
            else:
                current = candidate

        if current:
            result.append(current)

        # Reapply rules recursively
        final = []
        for chunk in result:
            final.extend(await self._finalize_chunk(chunk))
        return final

    async def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that were ≤ 50 tokens with next"""
        final_chunks, buffer = [], ""

        for chunk in chunks:
            if isinstance(chunk, tuple) and chunk[0] == "__SMALL__":
                if buffer:
                    buffer += " " + chunk[1]
                else:
                    buffer = chunk[1]
            else:
                if buffer:
                    merged = buffer + " " + chunk
                    tokens = await self.count_tokens(merged)
                    if tokens <= 400:
                        final_chunks.append(merged)
                    else:
                        final_chunks.append(buffer)
                        final_chunks.append(chunk)
                    buffer = ""
                else:
                    final_chunks.append(chunk)

        if buffer:  # leftover small piece
            final_chunks.append(buffer)

        return final_chunks

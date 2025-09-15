import re
import json
import pandas as pd

def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text):
    # Remove page numbers, repeated headers/footers, strange chars
    text = re.sub(r"\d+\s*اﻟﺠﺮﯾﺪة.*?\n", " ", text)   # remove repeated "الجريدة الرسمية ..." headers
    text = re.sub(r"\s+", " ", text)                   # collapse spaces
    return text.strip()

def split_into_chunks(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks

def save_chunks(chunks, json_path, excel_path):
    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Save Excel
    df = pd.DataFrame({"chunk": chunks})
    df.to_excel(excel_path, index=False, engine="openpyxl")

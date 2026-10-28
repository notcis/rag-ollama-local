import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import fitz  # PyMuPDF
import re
import unicodedata
from langchain_core.documents import Document
import numpy as np


DB_NAME = str(Path(__file__).parent.parent / "vector_db_2_2")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base_2_2")


embeddings = OllamaEmbeddings(model="embeddinggemma-2:570m")

# ตั้งเป็น False ถ้าไม่ต้องการ OCR

ENABLE_OCR_FALLBACK = True


_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\x00]")

# ช่องว่างที่ PDF มักแทรกก่อนสระบน/ล่าง/วรรณยุกต์/ไม้ไต่คู้ ฯลฯ

_SPACE_BEFORE_MARK = re.compile(r"\s+([\u0e31\u0e34-\u0e3a\u0e47-\u0e4e])")

_MULTI_NEWLINE = re.compile(r"\n{3,}")


_ocr_reader = None



def clean_thai_text(text: str) -> str:

    text = unicodedata.normalize("NFC", text)

    text = text.replace("\u0e4d\u0e32", "\u0e33")  # ํ + า -> ำ

    text = _ZERO_WIDTH.sub("", text)

    text = _SPACE_BEFORE_MARK.sub(r"\1", text)

    text = re.sub(r"[ \t]+", " ", text)

    return _MULTI_NEWLINE.sub("\n\n", text).strip()



def looks_garbled(text: str) -> bool:

    """ว่างเปล่า / มี (cid:xx) / มี � มากผิดปกติ -> น่าจะอ่านไม่ได้"""

    stripped = text.strip()

    if len(stripped) < 20:

        return True

    if "(cid:" in stripped:

        return True

    return stripped.count("\ufffd") / len(stripped) > 0.02



def ocr_page(page, dpi: int = 200) -> str:

    global _ocr_reader

    if _ocr_reader is None:

        import easyocr

        _ocr_reader = easyocr.Reader(["th", "en"], gpu=False)  # มี GPU ให้ใช้ gpu=True


    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    lines = _ocr_reader.readtext(img, detail=0, paragraph=True)

    return "\n".join(lines)



def load_pdf(path: str, doc_type: str) -> list[Document]:

    docs = []

    try:

        pdf = fitz.open(path)

    except Exception as e:

        print(f"[skip] เปิด PDF ไม่ได้: {path} ({e})")

        return docs


    with pdf:

        total = len(pdf)

        for i, page in enumerate(pdf):

            text = page.get_text("text", sort=True)

            method = "text"


            if looks_garbled(text) and ENABLE_OCR_FALLBACK:

                try:

                    text = ocr_page(page)

                    method = "ocr"

                except Exception as e:

                    print(f"[warn] OCR ไม่สำเร็จ {path} หน้า {i + 1}: {e}")


            text = clean_thai_text(text)

            if not text:

                continue


            docs.append(

                Document(

                    page_content=text,

                    metadata={

                        "source": path,

                        "page": i,

                        "total_pages": total,

                        "extract_method": method,

                        "doc_type": doc_type,

                    },

                )

            )

    return docs


def fetch_documents():

    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))

    documents = []


    # utf-8-sig จัดการ BOM ของไฟล์ไทยที่เซฟจาก Windows/Notepad

    text_patterns = ["**/*.txt", "**/*.text", "**/*.md"]

    text_kwargs = {"encoding": "utf-8-sig", "autodetect_encoding": True}


    for folder in folders:

        if not os.path.isdir(folder):

            continue

        doc_type = os.path.basename(folder)


        # ---- text / markdown ----

        for pattern in text_patterns:

            loader = DirectoryLoader(

                folder,

                glob=pattern,

                loader_cls=TextLoader,

                loader_kwargs=text_kwargs,

                silent_errors=True,

            )

            for doc in loader.load():

                doc.page_content = clean_thai_text(doc.page_content)

                doc.metadata["doc_type"] = doc_type

                documents.append(doc)


        # ---- pdf (รองรับ .pdf / .PDF) ----

        for pdf_path in Path(folder).rglob("*"):

            if pdf_path.suffix.lower() == ".pdf":

                documents.extend(load_pdf(str(pdf_path), doc_type))


    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")

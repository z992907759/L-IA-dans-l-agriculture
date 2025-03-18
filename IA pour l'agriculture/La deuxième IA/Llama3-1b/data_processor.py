import os
import json
import re
import fitz  # PyMuPDF 解析 PDF
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ✅ FAISS 存储路径
FAISS_INDEX_PATH = os.path.join(".", "database", "faiss_db")
DATA_FOLDER = os.path.join(".", "database", "knowledge_base")


def clean_text(text):
    """✅ 清理文本"""
    text = re.sub(r"\s{2,}", " ", text)  # 多余空格
    text = re.sub(r"\n{2,}", "\n", text)  # 多余换行
    return text.strip()


def process_pdf(file_path):
    """✅ 解析 PDF 并转换为问答格式"""
    print(f"📄 正在处理 PDF: {file_path}")
    doc = fitz.open(file_path)
    documents = []

    for page in doc:
        text = clean_text(page.get_text("text").strip())
        if len(text) > 100:  # 过滤过短文本
            question = f"What information does this text provide? {text[:100]}..."
            documents.append(Document(page_content=f"Question: {question}\nAnswer: {text}"))

    return documents


def process_txt(file_path):
    """✅ 解析 TXT 文件并转换为问答格式"""
    print(f"📄 正在处理 TXT: {file_path}")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = clean_text(f.read())

    if len(text) < 100:
        return []  # 过滤过短文本

    question = f"What information does this text provide? {text[:100]}..."
    return [Document(page_content=f"Question: {question}\nAnswer: {text}")]


def process_json(file_path):
    """✅ 解析 JSON 文件，并转换为问答格式"""
    print(f"📄 正在处理 JSON: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for entry in data:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()

        if question and answer:
            documents.append(Document(page_content=f"Question: {question}\nAnswer: {answer}"))

    return documents


def create_faiss_index():
    """✅ 重新处理所有 `knowledge_base/` 文件，生成 FAISS 索引"""
    print("🚀 正在创建 FAISS 索引...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={"device": device})

    all_docs = []
    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)
        if file.endswith(".pdf"):
            all_docs.extend(process_pdf(file_path))
        elif file.endswith(".txt"):
            all_docs.extend(process_txt(file_path))
        elif file.endswith(".json"):
            all_docs.extend(process_json(file_path))

    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
        split_docs = text_splitter.split_documents(all_docs)

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)

        print(f"✅ FAISS 索引创建完成，存入 {len(all_docs)} 条数据。")
    else:
        print("⚠️ 没有有效的内容可存入 FAISS。")


if __name__ == "__main__":
    create_faiss_index()

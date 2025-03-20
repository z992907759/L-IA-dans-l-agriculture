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
    """✅ 解析 TXT 文件并转换为问答格式，确保 FAISS 存储的是问答对"""
    print(f"📄 正在处理 TXT: {file_path}")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = clean_text(f.read())

    lines = text.split("\n")
    documents = []

    for i in range(0, len(lines) - 1, 2):  # 每两行作为 Q-A 对
        question = lines[i].strip()
        answer = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if question and answer and len(answer) > 30:  # 过滤掉无意义的短回答
            documents.append(Document(page_content=f"Question: {question}\nAnswer: {answer}"))

    return documents


def process_json(file_path):
    """✅ 解析 JSON 文件，并确保 `Question-Answer` 对的完整性"""
    print(f"📄 正在处理 JSON: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for entry in data:
        question = entry.get("input") or entry.get("question") or entry.get("QUESTION.question", "").strip()
        answer = entry.get("response") or entry.get("answers") or entry.get("ANSWER", "").strip()

        if question and answer and len(answer) > 30:  # 确保回答足够有信息量
            documents.append(Document(page_content=f"Question: {question}\nAnswer: {answer}"))

    return documents


def create_faiss_index():
    """✅ 重新处理所有 `knowledge_base/` 文件，生成 FAISS 索引"""
    print("🚀 正在创建 FAISS 索引...")
    device = "cpu"  # ✅ 强制使用 CPU，避免 MPS 错误
    embeddings = HuggingFaceEmbeddings(
        model_name="models/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

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
        print(f"✅ 已加载 {len(all_docs)} 条文档数据，准备创建 FAISS 索引...")

        # ✅ 改进切分策略
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=10)
        split_docs = text_splitter.split_documents(all_docs)

        # ✅ 确保 FAISS 存储的是清晰的问答数据
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)

        print(f"✅ FAISS 索引创建完成，存入 {len(split_docs)} 条数据。")
    else:
        print("⚠️ 没有有效的内容可存入 FAISS。")

def retrieve_faiss(query_text, top_k=1):
    """✅ 通过 FAISS 召回最相似的文本"""
    print(f"🔍 正在查询 FAISS: {query_text}")

    # ✅ 确保索引已加载
    print(f"📂 FAISS 索引路径: {FAISS_INDEX_PATH}")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        HuggingFaceEmbeddings(model_name="models/all-MiniLM-L6-v2",
                              model_kwargs={"device": "cpu"}),  # ✅ 强制使用 CPU
        allow_dangerous_deserialization=True
    )

    # ✅ 计算查询向量
    docs = vectorstore.similarity_search(query_text, k=top_k)

    if docs:
        return docs[0].page_content  # 返回最匹配的文本
    else:
        return "⚠️ 未找到相关文本！"




if __name__ == "__main__":
    create_faiss_index()

    # ✅ 检查 FAISS 召回结果
    test_query = "How can we prevent foodborne illness?"
    retrieved_text = retrieve_faiss(test_query)
    print(f"✅ 查询: {test_query}")
    print(f"🎯 FAISS 召回: {retrieved_text}")

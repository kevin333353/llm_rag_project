from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Dict
import shutil
from pathlib import Path
import logging
from rag_system import RAGSystem

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 導入您現有的 RAG 相關功能
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置上傳目錄
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"上傳目錄已創建: {UPLOAD_DIR}")

# 初始化 RAG 系統
rag_system = RAGSystem()

class Question(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"收到文件上傳請求: {file.filename}")
        
        # 檢查文件類型
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只接受 PDF 文件")
        
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"文件已保存到: {file_path}")
        
        # 處理文件
        rag_system.process_document(str(file_path))
        
        return {"message": "文件上傳成功"}
    except Exception as e:
        logger.error(f"文件上傳失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(question: Question):
    try:
        result = rag_system.query(question.question)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"查詢失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("啟動 FastAPI 服務器...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
# PDF 智能問答系統

這是一個基於 RAG (Retrieval-Augmented Generation) 技術的 PDF 智能問答系統。系統可以讓用戶上傳 PDF 文件，並基於文件內容進行智能問答。
![智慧問答](https://github.com/user-attachments/assets/8ab951d3-b0a1-4c59-973b-522f8bc46acf)


## 功能特點

- PDF 文件上傳與處理
- 基於文件內容的智能問答
- 現代化的 Web 界面
- 支持繁體中文
- 實時問答歷史記錄

## 技術棧

- 前端：React + Material-UI
- 後端：FastAPI
- 向量數據庫：FAISS
- 語言模型：Qwen2.5-7B (通過 Ollama)
- 文本嵌入：sentence-transformers

## 安裝與運行

### 前置需求

1. Python 3.8+
2. Node.js 14+
3. Ollama

### 安裝 Ollama

請參考 [Ollama 官方文檔](https://ollama.ai/) 安裝 Ollama，並下載 Qwen2.5-7B 模型：

```bash
ollama pull qwen2.5:7b
```

### 安裝依賴

1. 安裝 Python 依賴：
```bash
pip install -r requirements.txt
```

2. 安裝前端依賴：
```bash
cd frontend
npm install
```

### 運行系統

1. 啟動 Ollama 服務：
```bash
ollama serve
```

2. 啟動後端服務：
```bash
cd backend
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

3. 啟動前端服務：
```bash
cd frontend
npm start
```

4. 訪問系統：
打開瀏覽器訪問 http://localhost:3000

## 專案結構

```
pdf-qa-system/
├── backend/
│   ├── api.py              # FastAPI 後端服務
│   └── requirements.txt    # Python 依賴
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
├── data/
│   └── uploads/           # PDF 上傳目錄
└── README.md
```

## 使用說明

1. 上傳 PDF 文件：
   - 拖放 PDF 文件到上傳區域
   - 或點擊上傳區域選擇文件

2. 提問：
   - 在文本框中輸入問題
   - 點擊提交按鈕
   - 系統會基於 PDF 內容回答問題

## 注意事項

- 系統目前只支持 PDF 文件
- 請確保上傳的 PDF 文件可以正常打開
- 回答品質取決於 PDF 內容的品質和相關性

## 未來計劃

- [ ] 支持更多文件格式
- [ ] 添加文件管理功能
- [ ] 優化回答品質
- [ ] 添加用戶認證
- [ ] 支持多文件問答

## 授權

MIT License 

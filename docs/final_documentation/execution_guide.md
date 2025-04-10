# 專案執行指南

本文檔提供了智能文檔處理與問答系統的安裝和執行說明。

## 目錄結構

```
llm_rag_project/
├── code/
│   ├── rag_system/            # RAG系統實現
│   ├── fine_tuning/           # LLM微調模組
│   ├── quantization/          # 模型量化元件
│   ├── vector_db_integration/ # 向量資料庫整合
│   └── evaluation/            # 評估指標系統
├── data/
│   ├── sample_docs/           # 示例文檔
│   ├── processed/             # 處理後的文檔
│   ├── training_data/         # 微調數據
│   └── vector_store/          # 向量存儲
├── docs/                      # 文檔
├── requirements/              # 各模組依賴
└── requirements.txt           # 主要依賴
```

## 環境設置

### 1. 克隆專案（如果是從壓縮包獲取，則跳過此步驟）

```bash
git clone https://github.com/yourusername/llm-rag-project.git
cd llm-rag-project
```

### 2. 創建虛擬環境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 3. 安裝依賴

安裝主要依賴：

```bash
pip install -r requirements.txt
```

或者，根據需要安裝特定模組的依賴：

```bash
# 僅安裝RAG系統依賴
pip install -r requirements/rag_system_requirements.txt

# 僅安裝微調模組依賴
pip install -r requirements/fine_tuning_requirements.txt

# 僅安裝量化模組依賴
pip install -r requirements/quantization_requirements.txt

# 僅安裝向量資料庫依賴
pip install -r requirements/vector_db_requirements.txt

# 僅安裝評估系統依賴
pip install -r requirements/evaluation_requirements.txt
```

## 執行說明

### 1. RAG系統

```bash
cd code/rag_system
python main.py --documents_dir ../../data/sample_docs --query "什麼是RAG？"
```

參數說明：
- `--documents_dir`: 文檔目錄路徑
- `--query`: 查詢問題
- `--model_name`: (可選) LLM模型名稱，默認為"google/flan-t5-base"
- `--embedding_model`: (可選) 嵌入模型名稱，默認為"sentence-transformers/all-MiniLM-L6-v2"

### 2. LLM微調

```bash
cd code/fine_tuning
python fine_tuning.py --model_name meta-llama/Llama-3-8b-hf --dataset_path ../../data/training_data/dataset.json --output_dir ../../models/fine_tuned
```

參數說明：
- `--model_name`: 基礎模型名稱
- `--dataset_path`: 數據集路徑
- `--output_dir`: 輸出目錄
- `--lora_r`: (可選) LoRA秩，默認為16
- `--lora_alpha`: (可選) LoRA縮放，默認為32
- `--num_epochs`: (可選) 訓練輪數，默認為3

### 3. 模型量化

```bash
cd code/quantization
python model_quantization.py --model_name meta-llama/Llama-3-8b-hf --output_dir ../../models/quantized --bits 4 --quantization_type gptq
```

參數說明：
- `--model_name`: 模型名稱
- `--output_dir`: 輸出目錄
- `--bits`: 量化位元數，可選4或8
- `--quantization_type`: 量化類型，可選"gptq"或"awq"
- `--group_size`: (可選) 分組大小，默認為128

### 4. 向量資料庫整合

```bash
cd code/vector_db_integration
python vector_db_integration.py --db_type faiss --collection_name documents --documents_dir ../../data/sample_docs --query "什麼是向量資料庫？"
```

參數說明：
- `--db_type`: 向量資料庫類型，可選"faiss"、"chroma"、"weaviate"、"milvus"、"qdrant"
- `--collection_name`: 集合名稱
- `--documents_dir`: 文檔目錄路徑
- `--query`: 查詢問題
- `--persist_dir`: (可選) 持久化目錄，默認為"../../data/vector_store/{db_type}_{collection_name}"

### 5. 評估指標系統

```bash
cd code/evaluation
python main.py --mode evaluate --rag_system_path ../rag_system/main.py --output_dir ../../evaluation_results
```

參數說明：
- `--mode`: 運行模式，可選"example"、"evaluate"、"generate"
- `--rag_system_path`: RAG系統路徑
- `--output_dir`: 輸出目錄
- `--num_samples`: (可選) 樣本數量，默認為10

## 示例運行流程

以下是一個完整的示例運行流程：

1. 處理文檔並創建RAG系統：

```bash
cd code/rag_system
python main.py --documents_dir ../../data/sample_docs --save_vector_store
```

2. 使用RAG系統回答問題：

```bash
cd code/rag_system
python main.py --load_vector_store --query "什麼是RAG？"
```

3. 評估RAG系統性能：

```bash
cd code/evaluation
python main.py --mode evaluate --rag_system_path ../rag_system/main.py
```

## 注意事項

1. 對於需要大量計算資源的操作（如LLM微調和模型量化），建議使用具有GPU的環境。
2. 如果計算資源有限，可以使用較小的模型（如"google/flan-t5-base"）或量化後的模型。
3. 向量資料庫的選擇應根據數據規模和性能需求進行。對於小型應用，FAISS和Chroma足夠；對於大型應用，可以考慮Milvus或Qdrant。
4. 所有路徑都是相對於各模組目錄的相對路徑，請確保在正確的目錄中執行命令。

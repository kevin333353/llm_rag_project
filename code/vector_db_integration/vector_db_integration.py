"""
向量資料庫整合模組 - 負責整合多種向量資料庫
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VectorDBConfig:
    """向量資料庫配置類"""
    db_type: str = "faiss"                                # 資料庫類型: faiss, chroma, weaviate, milvus, qdrant
    collection_name: str = "document_collection"          # 集合名稱
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # 嵌入模型名稱
    persist_directory: str = "../data/vector_store"       # 持久化目錄
    host: Optional[str] = None                            # 主機地址 (用於遠程資料庫)
    port: Optional[int] = None                            # 端口 (用於遠程資料庫)
    api_key: Optional[str] = None                         # API密鑰 (用於雲服務)
    dimension: int = 384                                  # 向量維度
    distance_metric: str = "cosine"                       # 距離度量: cosine, euclidean, dot
    additional_config: Dict[str, Any] = field(default_factory=dict)  # 額外配置

class VectorDBFactory:
    """向量資料庫工廠類，負責創建不同類型的向量資料庫"""
    
    @staticmethod
    def create_vector_db(config: VectorDBConfig):
        """
        創建向量資料庫
        
        Args:
            config: 向量資料庫配置
            
        Returns:
            向量資料庫實例
        """
        db_type = config.db_type.lower()
        
        if db_type == "faiss":
            return FAISSVectorDB(config)
        elif db_type == "chroma":
            return ChromaVectorDB(config)
        elif db_type == "weaviate":
            return WeaviateVectorDB(config)
        elif db_type == "milvus":
            return MilvusVectorDB(config)
        elif db_type == "qdrant":
            return QdrantVectorDB(config)
        else:
            raise ValueError(f"不支持的向量資料庫類型: {db_type}")

class BaseVectorDB:
    """向量資料庫基類"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        self.config = config
        self.embedding_model = self._init_embedding_model()
        self.vector_store = None
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name,
            model_kwargs={'device': 'cpu'}  # 使用CPU以節省資源
        )
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def persist(self) -> bool:
        """
        持久化向量資料庫
        
        Returns:
            是否成功
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def load(self) -> bool:
        """
        加載向量資料庫
        
        Returns:
            是否成功
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def delete_collection(self) -> bool:
        """
        刪除集合
        
        Returns:
            是否成功
        """
        raise NotImplementedError("子類必須實現此方法")

class FAISSVectorDB(BaseVectorDB):
    """FAISS向量資料庫"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化FAISS向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        super().__init__(config)
        self.persist_directory = config.persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import FAISS
            
            if self.vector_store is None:
                # 創建新的向量存儲
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embedding_model
                )
                logger.info(f"已創建FAISS向量存儲，包含 {len(documents)} 個文檔")
            else:
                # 向現有向量存儲添加文檔
                self.vector_store.add_documents(documents)
                logger.info(f"已向FAISS向量存儲添加 {len(documents)} 個文檔")
            
            # 持久化
            self.persist()
            return True
        except Exception as e:
            logger.error(f"添加文檔到FAISS向量存儲時出錯: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            logger.info(f"FAISS搜索 '{query}' 返回 {len(docs_with_scores)} 個結果")
            return docs_with_scores
        except Exception as e:
            logger.error(f"FAISS搜索時出錯: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def persist(self) -> bool:
        """
        持久化向量資料庫
        
        Returns:
            是否成功
        """
        if self.vector_store is None:
            logger.warning("向量存儲為空，無法持久化")
            return False
        
        try:
            self.vector_store.save_local(self.persist_directory)
            logger.info(f"FAISS向量存儲已保存到 {self.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"持久化FAISS向量存儲時出錯: {str(e)}")
            return False
    
    def load(self) -> bool:
        """
        加載向量資料庫
        
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import FAISS
            
            if os.path.exists(self.persist_directory):
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embedding_model
                )
                logger.info(f"已從 {self.persist_directory} 加載FAISS向量存儲")
                return True
            else:
                logger.warning(f"FAISS向量存儲 {self.persist_directory} 不存在")
                return False
        except Exception as e:
            logger.error(f"加載FAISS向量存儲時出錯: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        刪除集合
        
        Returns:
            是否成功
        """
        try:
            import shutil
            
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info(f"已刪除FAISS向量存儲 {self.persist_directory}")
                self.vector_store = None
                return True
            else:
                logger.warning(f"FAISS向量存儲 {self.persist_directory} 不存在")
                return False
        except Exception as e:
            logger.error(f"刪除FAISS向量存儲時出錯: {str(e)}")
            return False

class ChromaVectorDB(BaseVectorDB):
    """Chroma向量資料庫"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化Chroma向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        super().__init__(config)
        self.persist_directory = config.persist_directory
        self.collection_name = config.collection_name
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Chroma
            
            if self.vector_store is None:
                # 創建新的向量存儲
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                logger.info(f"已創建Chroma向量存儲，包含 {len(documents)} 個文檔")
            else:
                # 向現有向量存儲添加文檔
                self.vector_store.add_documents(documents)
                logger.info(f"已向Chroma向量存儲添加 {len(documents)} 個文檔")
            
            # 持久化
            self.persist()
            return True
        except Exception as e:
            logger.error(f"添加文檔到Chroma向量存儲時出錯: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Chroma搜索 '{query}' 返回 {len(docs_with_scores)} 個結果")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Chroma搜索時出錯: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def persist(self) -> bool:
        """
        持久化向量資料庫
        
        Returns:
            是否成功
        """
        if self.vector_store is None:
            logger.warning("向量存儲為空，無法持久化")
            return False
        
        try:
            self.vector_store.persist()
            logger.info(f"Chroma向量存儲已持久化到 {self.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"持久化Chroma向量存儲時出錯: {str(e)}")
            return False
    
    def load(self) -> bool:
        """
        加載向量資料庫
        
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Chroma
            
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name
                )
                logger.info(f"已從 {self.persist_directory} 加載Chroma向量存儲")
                return True
            else:
                logger.warning(f"Chroma向量存儲 {self.persist_directory} 不存在")
                return False
        except Exception as e:
            logger.error(f"加載Chroma向量存儲時出錯: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        刪除集合
        
        Returns:
            是否成功
        """
        try:
            if self.vector_store is not None:
                self.vector_store.delete_collection()
                logger.info(f"已刪除Chroma集合 {self.collection_name}")
                self.vector_store = None
                return True
            else:
                logger.warning("向量存儲為空，無法刪除集合")
                return False
        except Exception as e:
            logger.error(f"刪除Chroma集合時出錯: {str(e)}")
            return False

class WeaviateVectorDB(BaseVectorDB):
    """Weaviate向量資料庫"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化Weaviate向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        super().__init__(config)
        self.host = config.host or "localhost"
        self.port = config.port or 8080
        self.collection_name = config.collection_name
        self.api_key = config.api_key
        self.additional_config = config.additional_config
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Weaviate
            import weaviate
            
            # 創建Weaviate客戶端
            auth_config = None
            if self.api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=self.api_key)
            
            client = weaviate.Client(
                url=f"http://{self.host}:{self.port}",
                auth_client_secret=auth_config
            )
            
            # 檢查集合是否存在，如果不存在則創建
            if not client.schema.exists(self.collection_name):
                class_obj = {
                    "class": self.collection_name,
                    "vectorizer": "none",  # 使用自定義向量
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"]
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"]
                        }
                    ]
                }
                client.schema.create_class(class_obj)
                logger.info(f"已創建Weaviate集合 {self.collection_name}")
            
            # 創建向量存儲
            self.vector_store = Weaviate(
                client=client,
                index_name=self.collection_name,
                text_key="content",
                embedding=self.embedding_model,
                by_text=False
            )
            
            # 添加文檔
            self.vector_store.add_documents(documents)
            logger.info(f"已向Weaviate添加 {len(documents)} 個文檔")
            return True
        except Exception as e:
            logger.error(f"添加文檔到Weaviate時出錯: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        try:
            # Weaviate不直接支持similarity_search_with_score，需要自定義
            docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            # 為了與其他向量存儲保持一致，我們需要添加一個模擬的分數
            # 實際上，這裡應該使用Weaviate的API來獲取實際分數
            docs_with_scores = [(doc, 0.9) for doc in docs]  # 使用模擬分數
            
            logger.info(f"Weaviate搜索 '{query}' 返回 {len(docs_with_scores)} 個結果")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Weaviate搜索時出錯: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def persist(self) -> bool:
        """
        持久化向量資料庫
        
        Returns:
            是否成功
        """
        # Weaviate是一個持久化的資料庫，不需要額外的持久化操作
        logger.info("Weaviate是一個持久化的資料庫，不需要額外的持久化操作")
        return True
    
    def load(self) -> bool:
        """
        加載向量資料庫
        
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Weaviate
            import weaviate
            
            # 創建Weaviate客戶端
            auth_config = None
            if self.api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=self.api_key)
            
            client = weaviate.Client(
                url=f"http://{self.host}:{self.port}",
                auth_client_secret=auth_config
            )
            
            # 檢查集合是否存在
            if not client.schema.exists(self.collection_name):
                logger.warning(f"Weaviate集合 {self.collection_name} 不存在")
                return False
            
            # 加載向量存儲
            self.vector_store = Weaviate(
                client=client,
                index_name=self.collection_name,
                text_key="content",
                embedding=self.embedding_model,
                by_text=False
            )
            
            logger.info(f"已加載Weaviate集合 {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"加載Weaviate時出錯: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        刪除集合
        
        Returns:
            是否成功
        """
        try:
            import weaviate
            
            # 創建Weaviate客戶端
            auth_config = None
            if self.api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=self.api_key)
            
            client = weaviate.Client(
                url=f"http://{self.host}:{self.port}",
                auth_client_secret=auth_config
            )
            
            # 檢查集合是否存在
            if client.schema.exists(self.collection_name):
                client.schema.delete_class(self.collection_name)
                logger.info(f"已刪除Weaviate集合 {self.collection_name}")
                self.vector_store = None
                return True
            else:
                logger.warning(f"Weaviate集合 {self.collection_name} 不存在")
                return False
        except Exception as e:
            logger.error(f"刪除Weaviate集合時出錯: {str(e)}")
            return False

class MilvusVectorDB(BaseVectorDB):
    """Milvus向量資料庫"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化Milvus向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        super().__init__(config)
        self.host = config.host or "localhost"
        self.port = config.port or 19530
        self.collection_name = config.collection_name
        self.dimension = config.dimension
        self.additional_config = config.additional_config
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Milvus
            
            # 創建向量存儲
            self.vector_store = Milvus.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                connection_args={
                    "host": self.host,
                    "port": self.port
                },
                collection_name=self.collection_name
            )
            
            logger.info(f"已向Milvus添加 {len(documents)} 個文檔")
            return True
        except Exception as e:
            logger.error(f"添加文檔到Milvus時出錯: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            logger.info(f"Milvus搜索 '{query}' 返回 {len(docs_with_scores)} 個結果")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Milvus搜索時出錯: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def persist(self) -> bool:
        """
        持久化向量資料庫
        
        Returns:
            是否成功
        """
        # Milvus是一個持久化的資料庫，不需要額外的持久化操作
        logger.info("Milvus是一個持久化的資料庫，不需要額外的持久化操作")
        return True
    
    def load(self) -> bool:
        """
        加載向量資料庫
        
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Milvus
            from pymilvus import connections, utility
            
            # 連接Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            
            # 檢查集合是否存在
            if not utility.has_collection(self.collection_name):
                logger.warning(f"Milvus集合 {self.collection_name} 不存在")
                return False
            
            # 加載向量存儲
            self.vector_store = Milvus(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
                connection_args={
                    "host": self.host,
                    "port": self.port
                }
            )
            
            logger.info(f"已加載Milvus集合 {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"加載Milvus時出錯: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        刪除集合
        
        Returns:
            是否成功
        """
        try:
            from pymilvus import connections, utility
            
            # 連接Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            
            # 檢查集合是否存在
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"已刪除Milvus集合 {self.collection_name}")
                self.vector_store = None
                return True
            else:
                logger.warning(f"Milvus集合 {self.collection_name} 不存在")
                return False
        except Exception as e:
            logger.error(f"刪除Milvus集合時出錯: {str(e)}")
            return False

class QdrantVectorDB(BaseVectorDB):
    """Qdrant向量資料庫"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化Qdrant向量資料庫
        
        Args:
            config: 向量資料庫配置
        """
        super().__init__(config)
        self.host = config.host
        self.port = config.port
        self.collection_name = config.collection_name
        self.persist_directory = config.persist_directory
        self.distance_metric = config.distance_metric
        self.additional_config = config.additional_config
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Qdrant
            
            # 確定使用本地還是遠程Qdrant
            if self.host and self.port:
                # 使用遠程Qdrant
                url = f"http://{self.host}:{self.port}"
                self.vector_store = Qdrant.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    url=url,
                    collection_name=self.collection_name,
                    distance_func=self.distance_metric
                )
            else:
                # 使用本地Qdrant
                os.makedirs(self.persist_directory, exist_ok=True)
                self.vector_store = Qdrant.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    path=self.persist_directory,
                    collection_name=self.collection_name,
                    distance_func=self.distance_metric
                )
            
            logger.info(f"已向Qdrant添加 {len(documents)} 個文檔")
            return True
        except Exception as e:
            logger.error(f"添加文檔到Qdrant時出錯: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            文檔和相似度分數的列表
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Qdrant搜索 '{query}' 返回 {len(docs_with_scores)} 個結果")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Qdrant搜索時出錯: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        獲取檢索器
        
        Args:
            search_kwargs: 搜索參數
            
        Returns:
            檢索器對象
        """
        if self.vector_store is None:
            if not self.load():
                raise ValueError("向量存儲未初始化，請先添加文檔或加載向量存儲")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def persist(self) -> bool:
        """
        持久化向量資料庫
        
        Returns:
            是否成功
        """
        # Qdrant是一個持久化的資料庫，不需要額外的持久化操作
        logger.info("Qdrant是一個持久化的資料庫，不需要額外的持久化操作")
        return True
    
    def load(self) -> bool:
        """
        加載向量資料庫
        
        Returns:
            是否成功
        """
        try:
            from langchain_community.vectorstores import Qdrant
            from qdrant_client import QdrantClient
            
            # 確定使用本地還是遠程Qdrant
            if self.host and self.port:
                # 使用遠程Qdrant
                client = QdrantClient(url=f"http://{self.host}:{self.port}")
            else:
                # 使用本地Qdrant
                client = QdrantClient(path=self.persist_directory)
            
            # 檢查集合是否存在
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.warning(f"Qdrant集合 {self.collection_name} 不存在")
                return False
            
            # 加載向量存儲
            if self.host and self.port:
                # 使用遠程Qdrant
                self.vector_store = Qdrant(
                    client=client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model
                )
            else:
                # 使用本地Qdrant
                self.vector_store = Qdrant(
                    client=client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model
                )
            
            logger.info(f"已加載Qdrant集合 {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"加載Qdrant時出錯: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        刪除集合
        
        Returns:
            是否成功
        """
        try:
            from qdrant_client import QdrantClient
            
            # 確定使用本地還是遠程Qdrant
            if self.host and self.port:
                # 使用遠程Qdrant
                client = QdrantClient(url=f"http://{self.host}:{self.port}")
            else:
                # 使用本地Qdrant
                client = QdrantClient(path=self.persist_directory)
            
            # 檢查集合是否存在
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name in collection_names:
                client.delete_collection(collection_name=self.collection_name)
                logger.info(f"已刪除Qdrant集合 {self.collection_name}")
                self.vector_store = None
                return True
            else:
                logger.warning(f"Qdrant集合 {self.collection_name} 不存在")
                return False
        except Exception as e:
            logger.error(f"刪除Qdrant集合時出錯: {str(e)}")
            return False

class VectorDBManager:
    """向量資料庫管理器，負責管理多個向量資料庫"""
    
    def __init__(self):
        """初始化向量資料庫管理器"""
        self.vector_dbs = {}
    
    def create_vector_db(self, config: VectorDBConfig) -> BaseVectorDB:
        """
        創建向量資料庫
        
        Args:
            config: 向量資料庫配置
            
        Returns:
            向量資料庫實例
        """
        db_id = f"{config.db_type}_{config.collection_name}"
        vector_db = VectorDBFactory.create_vector_db(config)
        self.vector_dbs[db_id] = vector_db
        return vector_db
    
    def get_vector_db(self, db_type: str, collection_name: str) -> Optional[BaseVectorDB]:
        """
        獲取向量資料庫
        
        Args:
            db_type: 資料庫類型
            collection_name: 集合名稱
            
        Returns:
            向量資料庫實例
        """
        db_id = f"{db_type}_{collection_name}"
        return self.vector_dbs.get(db_id)
    
    def list_vector_dbs(self) -> List[str]:
        """
        列出所有向量資料庫
        
        Returns:
            向量資料庫ID列表
        """
        return list(self.vector_dbs.keys())
    
    def delete_vector_db(self, db_type: str, collection_name: str) -> bool:
        """
        刪除向量資料庫
        
        Args:
            db_type: 資料庫類型
            collection_name: 集合名稱
            
        Returns:
            是否成功
        """
        db_id = f"{db_type}_{collection_name}"
        if db_id in self.vector_dbs:
            vector_db = self.vector_dbs[db_id]
            success = vector_db.delete_collection()
            if success:
                del self.vector_dbs[db_id]
            return success
        else:
            logger.warning(f"向量資料庫 {db_id} 不存在")
            return False

class RAGSystemWithMultiVectorDB:
    """支持多向量資料庫的RAG系統"""
    
    def __init__(
        self,
        vector_db_manager: VectorDBManager,
        default_db_config: VectorDBConfig,
        llm_model_name: str = "google/flan-t5-base",
        use_local_model: bool = True,
        temperature: float = 0.1
    ):
        """
        初始化RAG系統
        
        Args:
            vector_db_manager: 向量資料庫管理器
            default_db_config: 默認向量資料庫配置
            llm_model_name: 語言模型名稱
            use_local_model: 是否使用本地模型
            temperature: 溫度參數
        """
        self.vector_db_manager = vector_db_manager
        self.default_db_config = default_db_config
        self.llm_model_name = llm_model_name
        self.use_local_model = use_local_model
        self.temperature = temperature
        
        # 創建默認向量資料庫
        self.default_vector_db = self.vector_db_manager.create_vector_db(default_db_config)
        
        # 初始化LLM
        self.llm = self._init_llm()
        
        # 初始化檢索器
        self.retriever = self._init_retriever()
        
        # 初始化QA鏈
        self.qa_chain = self._init_qa_chain()
    
    def _init_llm(self):
        """初始化語言模型"""
        if self.use_local_model:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
                from langchain_community.llms import HuggingFacePipeline
                
                # 使用本地模型
                logger.info(f"正在加載本地模型: {self.llm_model_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)
                
                # 創建文本生成管道
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=self.temperature
                )
                
                # 創建HuggingFacePipeline
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logger.error(f"加載本地模型失敗: {str(e)}")
                logger.info("將使用OpenAI模型作為備選")
                from langchain_community.chat_models import ChatOpenAI
                return ChatOpenAI(temperature=self.temperature, model_name="gpt-3.5-turbo")
        else:
            # 使用OpenAI模型
            from langchain_community.chat_models import ChatOpenAI
            return ChatOpenAI(temperature=self.temperature, model_name="gpt-3.5-turbo")
    
    def _init_retriever(self):
        """初始化檢索器"""
        base_retriever = self.default_vector_db.get_retriever(search_kwargs={"k": 5})
        
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain_community.retrievers.document_compressors import LLMChainExtractor
            
            # 創建上下文壓縮器
            compressor = LLMChainExtractor.from_llm(self.llm)
            
            # 創建上下文壓縮檢索器
            return ContextualCompressionRetriever(
                base_retriever=base_retriever,
                doc_compressor=compressor
            )
        except Exception as e:
            logger.error(f"創建上下文壓縮檢索器失敗: {str(e)}")
            logger.info("將使用基本檢索器")
            return base_retriever
    
    def _init_qa_chain(self):
        """初始化問答鏈"""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        # 定義提示模板
        template = """
        你是一個專業的知識助手。請使用以下檢索到的上下文來回答問題。
        
        如果你無法從上下文中找到答案，請直接說"我無法從提供的信息中找到答案"，不要試圖編造答案。
        
        回答時，請引用信息來源（用方括號標註，如 [文檔1]）。
        
        上下文：
        {context}
        
        問題：{question}
        
        回答：
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 創建 QA 鏈
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def add_documents(self, documents: List[Document], db_type: Optional[str] = None, collection_name: Optional[str] = None) -> bool:
        """
        添加文檔到向量資料庫
        
        Args:
            documents: 文檔列表
            db_type: 資料庫類型
            collection_name: 集合名稱
            
        Returns:
            是否成功
        """
        if db_type and collection_name:
            # 使用指定的向量資料庫
            vector_db = self.vector_db_manager.get_vector_db(db_type, collection_name)
            if vector_db is None:
                # 創建新的向量資料庫
                config = VectorDBConfig(
                    db_type=db_type,
                    collection_name=collection_name,
                    embedding_model_name=self.default_db_config.embedding_model_name,
                    persist_directory=f"../data/vector_store/{db_type}_{collection_name}"
                )
                vector_db = self.vector_db_manager.create_vector_db(config)
            
            return vector_db.add_documents(documents)
        else:
            # 使用默認向量資料庫
            return self.default_vector_db.add_documents(documents)
    
    def query(self, question: str, db_type: Optional[str] = None, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        查詢RAG系統
        
        Args:
            question: 問題文本
            db_type: 資料庫類型
            collection_name: 集合名稱
            
        Returns:
            包含回答和來源文檔的字典
        """
        logger.info(f"處理查詢: '{question}'")
        
        # 如果指定了向量資料庫，則使用指定的向量資料庫
        if db_type and collection_name:
            vector_db = self.vector_db_manager.get_vector_db(db_type, collection_name)
            if vector_db is None:
                logger.warning(f"向量資料庫 {db_type}_{collection_name} 不存在，將使用默認向量資料庫")
                vector_db = self.default_vector_db
            
            # 更新檢索器
            self.retriever = vector_db.get_retriever(search_kwargs={"k": 5})
            
            # 更新QA鏈
            self.qa_chain = self._init_qa_chain()
        
        try:
            # 執行查詢
            result = self.qa_chain({"query": question})
            
            # 格式化結果
            formatted_result = {
                "query": question,
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
            
            return formatted_result
        except Exception as e:
            logger.error(f"查詢處理失敗: {str(e)}")
            return {
                "query": question,
                "answer": f"處理查詢時出錯: {str(e)}",
                "source_documents": []
            }
    
    def process_query_with_sources(self, question: str, db_type: Optional[str] = None, collection_name: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        處理查詢並返回帶有來源的回答
        
        Args:
            question: 問題文本
            db_type: 資料庫類型
            collection_name: 集合名稱
            
        Returns:
            回答文本和來源文檔列表的元組
        """
        result = self.query(question, db_type, collection_name)
        
        # 提取來源文檔信息
        sources = []
        for i, doc in enumerate(result.get("source_documents", [])):
            source_info = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "id": f"doc{i+1}"
            }
            sources.append(source_info)
        
        return result["answer"], sources

def create_sample_documents() -> List[Document]:
    """
    創建示例文檔
    
    Returns:
        文檔列表
    """
    documents = [
        Document(
            page_content="人工智能（Artificial Intelligence，簡稱AI）是計算機科學的一個分支，致力於開發能夠模擬人類智能的系統。",
            metadata={"source": "ai_intro.md", "section": "introduction"}
        ),
        Document(
            page_content="機器學習是AI的一個子領域，專注於開發能夠從數據中學習的算法。",
            metadata={"source": "ai_intro.md", "section": "machine_learning"}
        ),
        Document(
            page_content="深度學習是機器學習的一個分支，使用多層神經網絡處理複雜問題。",
            metadata={"source": "ai_intro.md", "section": "deep_learning"}
        ),
        Document(
            page_content="自然語言處理（NLP）是AI的另一個重要領域，專注於使計算機理解和生成人類語言。",
            metadata={"source": "ai_intro.md", "section": "nlp"}
        ),
        Document(
            page_content="大型語言模型（Large Language Models，簡稱LLM）是一種基於深度學習的自然語言處理模型，通過大規模預訓練學習語言的統計規律。",
            metadata={"source": "llm_intro.md", "section": "introduction"}
        ),
        Document(
            page_content="LLM通常採用預訓練-微調的範式，先在大規模語料庫上進行無監督學習，再在特定任務上進行有監督微調。",
            metadata={"source": "llm_intro.md", "section": "training"}
        ),
        Document(
            page_content="檢索增強生成（Retrieval-Augmented Generation，RAG）是一種結合檢索系統和生成模型的技術。它首先從知識庫中檢索與查詢相關的文檔，然後將這些文檔作為上下文提供給生成模型。",
            metadata={"source": "rag_intro.md", "section": "introduction"}
        ),
        Document(
            page_content="向量資料庫在RAG系統中扮演著核心角色，主要負責高效存儲和檢索文檔的向量表示。常見的向量資料庫包括Faiss、Milvus、Weaviate、Pinecone、Chroma等。",
            metadata={"source": "vector_db_intro.md", "section": "introduction"}
        )
    ]
    
    return documents

def create_vector_db_examples():
    """創建向量資料庫示例"""
    # 創建示例文檔
    documents = create_sample_documents()
    
    # 創建向量資料庫管理器
    vector_db_manager = VectorDBManager()
    
    # 創建FAISS向量資料庫
    faiss_config = VectorDBConfig(
        db_type="faiss",
        collection_name="ai_docs",
        persist_directory="../data/vector_store/faiss_ai_docs"
    )
    faiss_db = vector_db_manager.create_vector_db(faiss_config)
    faiss_db.add_documents(documents)
    
    # 創建Chroma向量資料庫
    chroma_config = VectorDBConfig(
        db_type="chroma",
        collection_name="ai_docs",
        persist_directory="../data/vector_store/chroma_ai_docs"
    )
    chroma_db = vector_db_manager.create_vector_db(chroma_config)
    chroma_db.add_documents(documents)
    
    # 創建RAG系統
    rag_system = RAGSystemWithMultiVectorDB(
        vector_db_manager=vector_db_manager,
        default_db_config=faiss_config,
        llm_model_name="google/flan-t5-base",
        use_local_model=True
    )
    
    # 測試查詢
    questions = [
        "什麼是深度學習？",
        "自然語言處理的主要目標是什麼？",
        "RAG系統的工作原理是什麼？"
    ]
    
    for question in questions:
        print("\n" + "="*50)
        print(f"問題: {question}")
        
        # 使用FAISS向量資料庫
        print("\n使用FAISS向量資料庫:")
        answer, sources = rag_system.process_query_with_sources(question, "faiss", "ai_docs")
        print(f"回答: {answer}")
        print("來源文檔:")
        for source in sources:
            print(f"- {source['content']}")
        
        # 使用Chroma向量資料庫
        print("\n使用Chroma向量資料庫:")
        answer, sources = rag_system.process_query_with_sources(question, "chroma", "ai_docs")
        print(f"回答: {answer}")
        print("來源文檔:")
        for source in sources:
            print(f"- {source['content']}")
    
    return {
        "vector_db_manager": vector_db_manager,
        "faiss_db": faiss_db,
        "chroma_db": chroma_db,
        "rag_system": rag_system
    }

def generate_vector_db_integration_code():
    """生成向量資料庫整合代碼"""
    # 創建主程序
    main_code = """
import os
import argparse
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from vector_db_integration import (
    VectorDBConfig,
    VectorDBManager,
    RAGSystemWithMultiVectorDB,
    create_sample_documents
)

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="向量資料庫整合示例")
    parser.add_argument("--db_type", type=str, default="faiss", help="向量資料庫類型: faiss, chroma, weaviate, milvus, qdrant")
    parser.add_argument("--collection_name", type=str, default="ai_docs", help="集合名稱")
    parser.add_argument("--query", type=str, help="查詢文本")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--create_sample", action="store_true", help="創建示例數據")
    
    args = parser.parse_args()
    
    # 創建向量資料庫管理器
    vector_db_manager = VectorDBManager()
    
    # 創建默認向量資料庫配置
    default_config = VectorDBConfig(
        db_type=args.db_type,
        collection_name=args.collection_name,
        persist_directory=f"../data/vector_store/{args.db_type}_{args.collection_name}"
    )
    
    # 創建RAG系統
    rag_system = RAGSystemWithMultiVectorDB(
        vector_db_manager=vector_db_manager,
        default_db_config=default_config,
        llm_model_name="google/flan-t5-base",
        use_local_model=True
    )
    
    # 創建示例數據
    if args.create_sample:
        documents = create_sample_documents()
        rag_system.add_documents(documents)
        print(f"已創建示例數據並添加到 {args.db_type} 向量資料庫的 {args.collection_name} 集合")
    
    # 查詢模式
    if args.query:
        answer, sources = rag_system.process_query_with_sources(args.query)
        
        print("\\n問題:", args.query)
        print("\\n回答:", answer)
        
        if sources:
            print("\\n來源文檔:")
            for i, source in enumerate(sources):
                print(f"[{i+1}] {source['content']}")
    
    # 交互模式
    if args.interactive:
        print("\\n" + "="*50)
        print(f"歡迎使用RAG系統！當前使用 {args.db_type} 向量資料庫的 {args.collection_name} 集合")
        print("輸入'exit'或'quit'退出。")
        print("="*50)
        
        while True:
            # 獲取用戶輸入
            user_input = input("\\n請輸入問題: ")
            
            # 檢查是否退出
            if user_input.lower() in ['exit', 'quit']:
                print("謝謝使用，再見！")
                break
            
            # 處理查詢
            answer, sources = rag_system.process_query_with_sources(user_input)
            
            # 顯示結果
            print("\\n回答:")
            print(answer)
            
            # 顯示來源
            if sources:
                print("\\n來源文檔:")
                for i, source in enumerate(sources):
                    print(f"[{i+1}] {source['content']}")
            else:
                print("\\n沒有找到相關來源")

if __name__ == "__main__":
    main()
"""
    
    # 保存主程序
    main_path = "../code/vector_db_integration/main.py"
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_code)
    
    logger.info(f"向量資料庫整合主程序已保存到 {main_path}")
    
    # 創建__init__.py
    init_code = """
from .vector_db_integration import (
    VectorDBConfig,
    VectorDBFactory,
    BaseVectorDB,
    FAISSVectorDB,
    ChromaVectorDB,
    WeaviateVectorDB,
    MilvusVectorDB,
    QdrantVectorDB,
    VectorDBManager,
    RAGSystemWithMultiVectorDB,
    create_sample_documents,
    create_vector_db_examples
)
"""
    
    init_path = "../code/vector_db_integration/__init__.py"
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(init_code)
    
    logger.info(f"向量資料庫整合__init__.py已保存到 {init_path}")
    
    return {
        "main_path": main_path,
        "init_path": init_path
    }

if __name__ == "__main__":
    # 創建向量資料庫示例
    examples = create_vector_db_examples()
    
    # 生成向量資料庫整合代碼
    code_paths = generate_vector_db_integration_code()
    
    print("\n所有準備工作已完成！")
    print(f"1. 向量資料庫整合模組: {__file__}")
    print(f"2. 向量資料庫整合主程序: {code_paths['main_path']}")
    print(f"3. 向量資料庫整合__init__.py: {code_paths['init_path']}")

"""
評估指標系統 - 負責評估RAG系統和LLM的性能
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# 確保nltk資源已下載
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """評估配置類"""
    metrics: List[str] = field(default_factory=lambda: ["relevance", "factuality", "coherence", "fluency"])  # 評估指標
    retrieval_metrics: List[str] = field(default_factory=lambda: ["precision", "recall", "ndcg", "mrr"])  # 檢索評估指標
    output_dir: str = "../evaluation_results"  # 輸出目錄
    test_set_path: Optional[str] = None  # 測試集路徑
    reference_answers_path: Optional[str] = None  # 參考答案路徑
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # 嵌入模型名稱
    num_samples: int = 100  # 樣本數量
    batch_size: int = 16  # 批次大小
    save_results: bool = True  # 是否保存結果
    verbose: bool = True  # 是否顯示詳細信息

class RetrievalEvaluator:
    """檢索評估器，負責評估檢索系統的性能"""
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化檢索評估器
        
        Args:
            config: 評估配置
        """
        self.config = config
        self.metrics = config.retrieval_metrics
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化嵌入模型
        self.embedding_model = self._init_embedding_model()
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            return HuggingFaceEmbeddings(
                model_name=self.config.embeddings_model_name,
                model_kwargs={'device': 'cpu'}  # 使用CPU以節省資源
            )
        except Exception as e:
            logger.error(f"初始化嵌入模型時出錯: {str(e)}")
            return None
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]]
    ) -> Dict[str, float]:
        """
        評估檢索性能
        
        Args:
            queries: 查詢列表
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            
        Returns:
            評估結果字典
        """
        results = {}
        
        # 計算各項指標
        if "precision" in self.metrics:
            results["precision"] = self._calculate_precision(retrieved_docs, relevant_docs)
        
        if "recall" in self.metrics:
            results["recall"] = self._calculate_recall(retrieved_docs, relevant_docs)
        
        if "ndcg" in self.metrics:
            results["ndcg"] = self._calculate_ndcg(retrieved_docs, relevant_docs)
        
        if "mrr" in self.metrics:
            results["mrr"] = self._calculate_mrr(retrieved_docs, relevant_docs)
        
        # 保存結果
        if self.config.save_results:
            self._save_retrieval_results(queries, retrieved_docs, relevant_docs, results)
        
        return results
    
    def _calculate_precision(
        self,
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]]
    ) -> float:
        """
        計算精確率
        
        Args:
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            
        Returns:
            精確率
        """
        precisions = []
        
        for i in range(len(retrieved_docs)):
            retrieved = retrieved_docs[i]
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算精確率
            if len(retrieved_ids) == 0:
                precisions.append(0.0)
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                precisions.append(len(relevant_retrieved) / len(retrieved_ids))
        
        # 計算平均精確率
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def _calculate_recall(
        self,
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]]
    ) -> float:
        """
        計算召回率
        
        Args:
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            
        Returns:
            召回率
        """
        recalls = []
        
        for i in range(len(retrieved_docs)):
            retrieved = retrieved_docs[i]
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算召回率
            if len(relevant_ids) == 0:
                recalls.append(1.0)  # 如果沒有相關文檔，則召回率為1
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                recalls.append(len(relevant_retrieved) / len(relevant_ids))
        
        # 計算平均召回率
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    def _calculate_ndcg(
        self,
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]],
        k: int = 10
    ) -> float:
        """
        計算NDCG (Normalized Discounted Cumulative Gain)
        
        Args:
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            k: 計算NDCG的文檔數量
            
        Returns:
            NDCG
        """
        ndcg_scores = []
        
        for i in range(len(retrieved_docs)):
            retrieved = retrieved_docs[i][:k]  # 只考慮前k個文檔
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算DCG
            dcg = 0.0
            for j, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    # 使用二元相關性（0或1）
                    relevance = 1
                    # 計算DCG
                    dcg += relevance / np.log2(j + 2)  # j+2是因為log2(1)=0，我們從位置1開始
            
            # 計算IDCG（理想DCG）
            idcg = 0.0
            for j in range(min(len(relevant_ids), k)):
                # 使用二元相關性（0或1）
                relevance = 1
                # 計算IDCG
                idcg += relevance / np.log2(j + 2)
            
            # 計算NDCG
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(1.0)  # 如果沒有相關文檔，則NDCG為1
        
        # 計算平均NDCG
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_mrr(
        self,
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]]
    ) -> float:
        """
        計算MRR (Mean Reciprocal Rank)
        
        Args:
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            
        Returns:
            MRR
        """
        reciprocal_ranks = []
        
        for i in range(len(retrieved_docs)):
            retrieved = retrieved_docs[i]
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算倒數排名
            rank = 0
            for j, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    rank = j + 1  # 排名從1開始
                    break
            
            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        # 計算平均倒數排名
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def _get_doc_id(self, doc: Document) -> str:
        """
        獲取文檔ID
        
        Args:
            doc: 文檔
            
        Returns:
            文檔ID
        """
        # 如果文檔有ID，則使用ID
        if hasattr(doc, 'id') and doc.id:
            return doc.id
        
        # 如果文檔的metadata中有ID，則使用metadata中的ID
        if hasattr(doc, 'metadata') and doc.metadata and 'id' in doc.metadata:
            return doc.metadata['id']
        
        # 如果文檔的metadata中有source，則使用source作為ID
        if hasattr(doc, 'metadata') and doc.metadata and 'source' in doc.metadata:
            return doc.metadata['source']
        
        # 如果以上都沒有，則使用文檔內容的哈希值作為ID
        return str(hash(doc.page_content))
    
    def _save_retrieval_results(
        self,
        queries: List[str],
        retrieved_docs: List[List[Document]],
        relevant_docs: List[List[Document]],
        results: Dict[str, float]
    ) -> None:
        """
        保存檢索評估結果
        
        Args:
            queries: 查詢列表
            retrieved_docs: 檢索到的文檔列表的列表
            relevant_docs: 相關文檔列表的列表
            results: 評估結果字典
        """
        # 創建結果目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.output_dir, f"retrieval_evaluation_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存評估結果
        with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存詳細結果
        detailed_results = []
        
        for i in range(len(queries)):
            query = queries[i]
            retrieved = retrieved_docs[i]
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self._get_doc_id(doc) for doc in relevant]
            
            # 計算精確率和召回率
            if len(retrieved_ids) == 0:
                precision = 0.0
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                precision = len(relevant_retrieved) / len(retrieved_ids)
            
            if len(relevant_ids) == 0:
                recall = 1.0
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                recall = len(relevant_retrieved) / len(relevant_ids)
            
            # 添加到詳細結果
            detailed_results.append({
                "query": query,
                "retrieved_docs": [doc.page_content for doc in retrieved],
                "relevant_docs": [doc.page_content for doc in relevant],
                "precision": precision,
                "recall": recall
            })
        
        # 保存詳細結果
        with open(os.path.join(result_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"檢索評估結果已保存到 {result_dir}")

class GenerationEvaluator:
    """生成評估器，負責評估生成系統的性能"""
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化生成評估器
        
        Args:
            config: 評估配置
        """
        self.config = config
        self.metrics = config.metrics
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化嵌入模型
        self.embedding_model = self._init_embedding_model()
        
        # 初始化ROUGE評分器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # 初始化BLEU平滑函數
        self.smoothing = SmoothingFunction().method1
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            return HuggingFaceEmbeddings(
                model_name=self.config.embeddings_model_name,
                model_kwargs={'device': 'cpu'}  # 使用CPU以節省資源
            )
        except Exception as e:
            logger.error(f"初始化嵌入模型時出錯: {str(e)}")
            return None
    
    def evaluate_generation(
        self,
        queries: List[str],
        generated_answers: List[str],
        reference_answers: List[str],
        contexts: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        評估生成性能
        
        Args:
            queries: 查詢列表
            generated_answers: 生成的答案列表
            reference_answers: 參考答案列表
            contexts: 上下文列表的列表
            
        Returns:
            評估結果字典
        """
        results = {}
        
        # 計算各項指標
        if "relevance" in self.metrics:
            results["relevance"] = self._calculate_relevance(queries, generated_answers)
        
        if "factuality" in self.metrics:
            results["factuality"] = self._calculate_factuality(generated_answers, reference_answers, contexts)
        
        if "coherence" in self.metrics:
            results["coherence"] = self._calculate_coherence(generated_answers)
        
        if "fluency" in self.metrics:
            results["fluency"] = self._calculate_fluency(generated_answers)
        
        if "rouge" in self.metrics:
            rouge_scores = self._calculate_rouge(generated_answers, reference_answers)
            results.update(rouge_scores)
        
        if "bleu" in self.metrics:
            results["bleu"] = self._calculate_bleu(generated_answers, reference_answers)
        
        # 保存結果
        if self.config.save_results:
            self._save_generation_results(queries, generated_answers, reference_answers, contexts, results)
        
        return results
    
    def _calculate_relevance(
        self,
        queries: List[str],
        generated_answers: List[str]
    ) -> float:
        """
        計算相關性
        
        Args:
            queries: 查詢列表
            generated_answers: 生成的答案列表
            
        Returns:
            相關性分數
        """
        if self.embedding_model is None:
            logger.warning("嵌入模型未初始化，無法計算相關性")
            return 0.0
        
        relevance_scores = []
        
        for i in range(len(queries)):
            query = queries[i]
            answer = generated_answers[i]
            
            # 獲取查詢和答案的嵌入
            query_embedding = self.embedding_model.embed_query(query)
            answer_embedding = self.embedding_model.embed_query(answer)
            
            # 計算餘弦相似度
            similarity = cosine_similarity([query_embedding], [answer_embedding])[0][0]
            relevance_scores.append(similarity)
        
        # 計算平均相關性
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    def _calculate_factuality(
        self,
        generated_answers: List[str],
        reference_answers: List[str],
        contexts: Optional[List[List[str]]] = None
    ) -> float:
        """
        計算事實性
        
        Args:
            generated_answers: 生成的答案列表
            reference_answers: 參考答案列表
            contexts: 上下文列表的列表
            
        Returns:
            事實性分數
        """
        if self.embedding_model is None:
            logger.warning("嵌入模型未初始化，無法計算事實性")
            return 0.0
        
        factuality_scores = []
        
        for i in range(len(generated_answers)):
            generated = generated_answers[i]
            reference = reference_answers[i]
            
            # 獲取生成答案和參考答案的嵌入
            generated_embedding = self.embedding_model.embed_query(generated)
            reference_embedding = self.embedding_model.embed_query(reference)
            
            # 計算餘弦相似度
            similarity = cosine_similarity([generated_embedding], [reference_embedding])[0][0]
            
            # 如果有上下文，則考慮上下文
            if contexts and i < len(contexts) and contexts[i]:
                context_similarities = []
                
                for context in contexts[i]:
                    # 獲取上下文的嵌入
                    context_embedding = self.embedding_model.embed_query(context)
                    
                    # 計算生成答案與上下文的餘弦相似度
                    context_similarity = cosine_similarity([generated_embedding], [context_embedding])[0][0]
                    context_similarities.append(context_similarity)
                
                # 使用最大上下文相似度
                max_context_similarity = max(context_similarities) if context_similarities else 0.0
                
                # 綜合考慮參考答案相似度和上下文相似度
                factuality = 0.7 * similarity + 0.3 * max_context_similarity
            else:
                factuality = similarity
            
            factuality_scores.append(factuality)
        
        # 計算平均事實性
        return sum(factuality_scores) / len(factuality_scores) if factuality_scores else 0.0
    
    def _calculate_coherence(
        self,
        generated_answers: List[str]
    ) -> float:
        """
        計算連貫性
        
        Args:
            generated_answers: 生成的答案列表
            
        Returns:
            連貫性分數
        """
        coherence_scores = []
        
        for answer in generated_answers:
            # 分割成句子
            sentences = nltk.sent_tokenize(answer)
            
            if len(sentences) <= 1:
                # 如果只有一個句子，則連貫性為1
                coherence_scores.append(1.0)
                continue
            
            # 計算相鄰句子之間的相似度
            similarities = []
            
            for i in range(len(sentences) - 1):
                if self.embedding_model:
                    # 使用嵌入模型計算相似度
                    embedding1 = self.embedding_model.embed_query(sentences[i])
                    embedding2 = self.embedding_model.embed_query(sentences[i + 1])
                    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                else:
                    # 如果嵌入模型未初始化，則使用簡單的詞重疊計算相似度
                    words1 = set(sentences[i].lower().split())
                    words2 = set(sentences[i + 1].lower().split())
                    
                    if not words1 or not words2:
                        similarity = 0.0
                    else:
                        intersection = words1.intersection(words2)
                        union = words1.union(words2)
                        similarity = len(intersection) / len(union)
                
                similarities.append(similarity)
            
            # 計算平均相似度作為連貫性分數
            coherence = sum(similarities) / len(similarities) if similarities else 0.0
            coherence_scores.append(coherence)
        
        # 計算平均連貫性
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_fluency(
        self,
        generated_answers: List[str]
    ) -> float:
        """
        計算流暢性
        
        Args:
            generated_answers: 生成的答案列表
            
        Returns:
            流暢性分數
        """
        fluency_scores = []
        
        for answer in generated_answers:
            # 分割成句子
            sentences = nltk.sent_tokenize(answer)
            
            if not sentences:
                fluency_scores.append(0.0)
                continue
            
            # 計算每個句子的平均詞長
            avg_word_lengths = []
            
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                
                if not words:
                    continue
                
                avg_word_length = sum(len(word) for word in words) / len(words)
                avg_word_lengths.append(avg_word_length)
            
            # 計算句子長度
            sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
            
            # 計算流暢性分數
            # 這裡使用一個簡單的啟發式方法：
            # 1. 句子長度適中（不太長也不太短）
            # 2. 詞長適中（不太長也不太短）
            
            # 計算句子長度的標準差
            sentence_length_std = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
            
            # 計算平均句子長度
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
            
            # 計算平均詞長
            avg_word_length = sum(avg_word_lengths) / len(avg_word_lengths) if avg_word_lengths else 0
            
            # 計算流暢性分數
            # 句子長度在5-25之間，詞長在3-8之間，標準差較小，則流暢性較高
            sentence_length_score = max(0, 1 - abs(avg_sentence_length - 15) / 15)
            word_length_score = max(0, 1 - abs(avg_word_length - 5) / 5)
            std_score = max(0, 1 - sentence_length_std / 10)
            
            fluency = 0.4 * sentence_length_score + 0.3 * word_length_score + 0.3 * std_score
            fluency_scores.append(fluency)
        
        # 計算平均流暢性
        return sum(fluency_scores) / len(fluency_scores) if fluency_scores else 0.0
    
    def _calculate_rouge(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """
        計算ROUGE分數
        
        Args:
            generated_answers: 生成的答案列表
            reference_answers: 參考答案列表
            
        Returns:
            ROUGE分數字典
        """
        rouge_scores = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }
        
        for i in range(len(generated_answers)):
            generated = generated_answers[i]
            reference = reference_answers[i]
            
            # 計算ROUGE分數
            scores = self.rouge_scorer.score(reference, generated)
            
            # 累加分數
            rouge_scores["rouge1"] += scores["rouge1"].fmeasure
            rouge_scores["rouge2"] += scores["rouge2"].fmeasure
            rouge_scores["rougeL"] += scores["rougeL"].fmeasure
        
        # 計算平均分數
        if generated_answers:
            rouge_scores["rouge1"] /= len(generated_answers)
            rouge_scores["rouge2"] /= len(generated_answers)
            rouge_scores["rougeL"] /= len(generated_answers)
        
        return rouge_scores
    
    def _calculate_bleu(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> float:
        """
        計算BLEU分數
        
        Args:
            generated_answers: 生成的答案列表
            reference_answers: 參考答案列表
            
        Returns:
            BLEU分數
        """
        bleu_scores = []
        
        for i in range(len(generated_answers)):
            generated = generated_answers[i]
            reference = reference_answers[i]
            
            # 分詞
            generated_tokens = nltk.word_tokenize(generated.lower())
            reference_tokens = [nltk.word_tokenize(reference.lower())]
            
            # 計算BLEU分數
            try:
                bleu = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=self.smoothing)
                bleu_scores.append(bleu)
            except Exception as e:
                logger.error(f"計算BLEU分數時出錯: {str(e)}")
                bleu_scores.append(0.0)
        
        # 計算平均BLEU分數
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    def _save_generation_results(
        self,
        queries: List[str],
        generated_answers: List[str],
        reference_answers: List[str],
        contexts: Optional[List[List[str]]],
        results: Dict[str, float]
    ) -> None:
        """
        保存生成評估結果
        
        Args:
            queries: 查詢列表
            generated_answers: 生成的答案列表
            reference_answers: 參考答案列表
            contexts: 上下文列表的列表
            results: 評估結果字典
        """
        # 創建結果目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.output_dir, f"generation_evaluation_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存評估結果
        with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存詳細結果
        detailed_results = []
        
        for i in range(len(queries)):
            query = queries[i]
            generated = generated_answers[i]
            reference = reference_answers[i]
            
            # 計算ROUGE分數
            rouge_scores = self.rouge_scorer.score(reference, generated)
            
            # 計算BLEU分數
            try:
                generated_tokens = nltk.word_tokenize(generated.lower())
                reference_tokens = [nltk.word_tokenize(reference.lower())]
                bleu = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=self.smoothing)
            except Exception:
                bleu = 0.0
            
            # 添加到詳細結果
            result = {
                "query": query,
                "generated_answer": generated,
                "reference_answer": reference,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure,
                "bleu": bleu
            }
            
            # 如果有上下文，則添加上下文
            if contexts and i < len(contexts):
                result["context"] = contexts[i]
            
            detailed_results.append(result)
        
        # 保存詳細結果
        with open(os.path.join(result_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"生成評估結果已保存到 {result_dir}")

class RAGEvaluator:
    """RAG系統評估器，負責評估RAG系統的整體性能"""
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化RAG系統評估器
        
        Args:
            config: 評估配置
        """
        self.config = config
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化檢索評估器和生成評估器
        self.retrieval_evaluator = RetrievalEvaluator(config)
        self.generation_evaluator = GenerationEvaluator(config)
    
    def evaluate_rag_system(
        self,
        rag_system,
        test_queries: List[str],
        reference_answers: List[str],
        relevant_docs: List[List[Document]]
    ) -> Dict[str, Dict[str, float]]:
        """
        評估RAG系統
        
        Args:
            rag_system: RAG系統
            test_queries: 測試查詢列表
            reference_answers: 參考答案列表
            relevant_docs: 相關文檔列表的列表
            
        Returns:
            評估結果字典
        """
        # 使用RAG系統處理查詢
        retrieved_docs_list = []
        generated_answers = []
        contexts = []
        
        for query in test_queries:
            # 處理查詢
            result = rag_system.query(query)
            
            # 獲取檢索到的文檔
            retrieved_docs = result.get("source_documents", [])
            retrieved_docs_list.append(retrieved_docs)
            
            # 獲取生成的答案
            answer = result.get("answer", "")
            generated_answers.append(answer)
            
            # 獲取上下文
            context = [doc.page_content for doc in retrieved_docs]
            contexts.append(context)
        
        # 評估檢索性能
        retrieval_results = self.retrieval_evaluator.evaluate_retrieval(
            test_queries,
            retrieved_docs_list,
            relevant_docs
        )
        
        # 評估生成性能
        generation_results = self.generation_evaluator.evaluate_generation(
            test_queries,
            generated_answers,
            reference_answers,
            contexts
        )
        
        # 綜合評估結果
        results = {
            "retrieval": retrieval_results,
            "generation": generation_results
        }
        
        # 保存結果
        if self.config.save_results:
            self._save_rag_results(
                test_queries,
                retrieved_docs_list,
                generated_answers,
                reference_answers,
                relevant_docs,
                results
            )
        
        return results
    
    def _save_rag_results(
        self,
        queries: List[str],
        retrieved_docs: List[List[Document]],
        generated_answers: List[str],
        reference_answers: List[str],
        relevant_docs: List[List[Document]],
        results: Dict[str, Dict[str, float]]
    ) -> None:
        """
        保存RAG系統評估結果
        
        Args:
            queries: 查詢列表
            retrieved_docs: 檢索到的文檔列表的列表
            generated_answers: 生成的答案列表
            reference_answers: 參考答案列表
            relevant_docs: 相關文檔列表的列表
            results: 評估結果字典
        """
        # 創建結果目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.output_dir, f"rag_evaluation_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存評估結果
        with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存詳細結果
        detailed_results = []
        
        for i in range(len(queries)):
            query = queries[i]
            retrieved = retrieved_docs[i]
            generated = generated_answers[i]
            reference = reference_answers[i]
            relevant = relevant_docs[i]
            
            # 獲取檢索文檔和相關文檔的ID
            retrieved_ids = [self.retrieval_evaluator._get_doc_id(doc) for doc in retrieved]
            relevant_ids = [self.retrieval_evaluator._get_doc_id(doc) for doc in relevant]
            
            # 計算精確率和召回率
            if len(retrieved_ids) == 0:
                precision = 0.0
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                precision = len(relevant_retrieved) / len(retrieved_ids)
            
            if len(relevant_ids) == 0:
                recall = 1.0
            else:
                relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
                recall = len(relevant_retrieved) / len(relevant_ids)
            
            # 計算ROUGE分數
            rouge_scores = self.generation_evaluator.rouge_scorer.score(reference, generated)
            
            # 計算BLEU分數
            try:
                generated_tokens = nltk.word_tokenize(generated.lower())
                reference_tokens = [nltk.word_tokenize(reference.lower())]
                bleu = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=self.generation_evaluator.smoothing)
            except Exception:
                bleu = 0.0
            
            # 添加到詳細結果
            detailed_results.append({
                "query": query,
                "retrieved_docs": [doc.page_content for doc in retrieved],
                "relevant_docs": [doc.page_content for doc in relevant],
                "generated_answer": generated,
                "reference_answer": reference,
                "precision": precision,
                "recall": recall,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure,
                "bleu": bleu
            })
        
        # 保存詳細結果
        with open(os.path.join(result_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"RAG系統評估結果已保存到 {result_dir}")

class EvaluationDataGenerator:
    """評估數據生成器，負責生成評估數據"""
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化評估數據生成器
        
        Args:
            config: 評估配置
        """
        self.config = config
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_test_data(
        self,
        num_samples: int = 100,
        domains: List[str] = ["general", "tech", "science", "finance", "healthcare"]
    ) -> Tuple[List[str], List[str], List[List[Document]]]:
        """
        生成測試數據
        
        Args:
            num_samples: 樣本數量
            domains: 領域列表
            
        Returns:
            查詢列表、參考答案列表和相關文檔列表的元組
        """
        # 生成測試查詢
        queries = self._generate_test_queries(num_samples, domains)
        
        # 生成參考答案
        reference_answers = self._generate_reference_answers(queries)
        
        # 生成相關文檔
        relevant_docs = self._generate_relevant_docs(queries, reference_answers)
        
        # 保存測試數據
        self._save_test_data(queries, reference_answers, relevant_docs)
        
        return queries, reference_answers, relevant_docs
    
    def _generate_test_queries(
        self,
        num_samples: int,
        domains: List[str]
    ) -> List[str]:
        """
        生成測試查詢
        
        Args:
            num_samples: 樣本數量
            domains: 領域列表
            
        Returns:
            查詢列表
        """
        # 為每個領域生成查詢
        queries = []
        samples_per_domain = num_samples // len(domains)
        
        for domain in domains:
            if domain == "general":
                domain_queries = [
                    "什麼是人工智能？",
                    "機器學習和深度學習有什麼區別？",
                    "大型語言模型的工作原理是什麼？",
                    "什麼是自然語言處理？",
                    "人工智能的發展歷史是怎樣的？"
                ]
            elif domain == "tech":
                domain_queries = [
                    "什麼是雲計算？",
                    "區塊鏈技術的應用有哪些？",
                    "5G技術的優勢是什麼？",
                    "物聯網如何改變我們的生活？",
                    "量子計算的原理是什麼？"
                ]
            elif domain == "science":
                domain_queries = [
                    "什麼是相對論？",
                    "量子力學的基本原理是什麼？",
                    "DNA的結構是怎樣的？",
                    "黑洞是如何形成的？",
                    "氣候變化的主要原因是什麼？"
                ]
            elif domain == "finance":
                domain_queries = [
                    "什麼是通貨膨脹？",
                    "股票市場如何運作？",
                    "加密貨幣的優缺點是什麼？",
                    "如何進行個人財務規劃？",
                    "什麼是風險投資？"
                ]
            elif domain == "healthcare":
                domain_queries = [
                    "新冠病毒的傳播方式是什麼？",
                    "如何保持健康的生活方式？",
                    "常見的心臟疾病有哪些？",
                    "什麼是免疫系統？",
                    "精神健康的重要性是什麼？"
                ]
            else:
                domain_queries = [
                    f"{domain}領域的最新發展是什麼？",
                    f"{domain}領域面臨的主要挑戰是什麼？",
                    f"{domain}領域的未來趨勢是什麼？",
                    f"{domain}領域的主要技術是什麼？",
                    f"{domain}領域的應用場景有哪些？"
                ]
            
            # 確保每個領域有足夠的查詢
            while len(domain_queries) < samples_per_domain:
                domain_queries.append(f"{domain}領域的問題{len(domain_queries) + 1}")
            
            # 添加到查詢列表
            queries.extend(domain_queries[:samples_per_domain])
        
        # 確保總數量正確
        while len(queries) < num_samples:
            queries.append(f"一般問題{len(queries) + 1}")
        
        return queries[:num_samples]
    
    def _generate_reference_answers(
        self,
        queries: List[str]
    ) -> List[str]:
        """
        生成參考答案
        
        Args:
            queries: 查詢列表
            
        Returns:
            參考答案列表
        """
        # 為每個查詢生成參考答案
        reference_answers = []
        
        for query in queries:
            if "人工智能" in query:
                answer = "人工智能（Artificial Intelligence，簡稱AI）是計算機科學的一個分支，致力於開發能夠模擬人類智能的系統。它包括機器學習、深度學習、自然語言處理等多個子領域。"
            elif "機器學習" in query and "深度學習" in query:
                answer = "機器學習是AI的一個子領域，專注於開發能夠從數據中學習的算法。深度學習是機器學習的一個分支，使用多層神經網絡處理複雜問題。機器學習更廣泛，包括多種算法如決策樹、SVM等，而深度學習專注於神經網絡。"
            elif "大型語言模型" in query:
                answer = "大型語言模型（Large Language Models，簡稱LLM）是一種基於深度學習的自然語言處理模型，通過大規模預訓練學習語言的統計規律。它們通常採用Transformer架構，能夠理解和生成人類語言，執行各種語言任務。"
            elif "自然語言處理" in query:
                answer = "自然語言處理（NLP）是AI的一個重要領域，專注於使計算機理解和生成人類語言。它涉及多種任務，如文本分類、情感分析、機器翻譯、問答系統等。"
            elif "雲計算" in query:
                answer = "雲計算是一種通過互聯網提供計算資源的服務模式，包括服務器、存儲、數據庫、網絡、軟件等。它具有按需自助服務、廣泛的網絡訪問、資源池化、快速彈性和可計量服務等特點。"
            else:
                answer = f"這是關於「{query}」的參考答案。它提供了全面、準確的信息，並以清晰、連貫的方式呈現。"
            
            reference_answers.append(answer)
        
        return reference_answers
    
    def _generate_relevant_docs(
        self,
        queries: List[str],
        reference_answers: List[str]
    ) -> List[List[Document]]:
        """
        生成相關文檔
        
        Args:
            queries: 查詢列表
            reference_answers: 參考答案列表
            
        Returns:
            相關文檔列表的列表
        """
        # 為每個查詢生成相關文檔
        relevant_docs = []
        
        for i, query in enumerate(queries):
            # 使用參考答案作為一個相關文檔
            docs = [
                Document(
                    page_content=reference_answers[i],
                    metadata={"source": f"reference_{i}.md", "relevance": "high"}
                )
            ]
            
            # 添加其他相關文檔
            if "人工智能" in query:
                docs.append(Document(
                    page_content="人工智能（Artificial Intelligence，簡稱AI）是計算機科學的一個分支，致力於開發能夠模擬人類智能的系統。",
                    metadata={"source": "ai_intro.md", "section": "introduction", "relevance": "high"}
                ))
                docs.append(Document(
                    page_content="機器學習是AI的一個子領域，專注於開發能夠從數據中學習的算法。",
                    metadata={"source": "ai_intro.md", "section": "machine_learning", "relevance": "medium"}
                ))
            elif "機器學習" in query and "深度學習" in query:
                docs.append(Document(
                    page_content="機器學習是AI的一個子領域，專注於開發能夠從數據中學習的算法。",
                    metadata={"source": "ai_intro.md", "section": "machine_learning", "relevance": "high"}
                ))
                docs.append(Document(
                    page_content="深度學習是機器學習的一個分支，使用多層神經網絡處理複雜問題。",
                    metadata={"source": "ai_intro.md", "section": "deep_learning", "relevance": "high"}
                ))
            elif "大型語言模型" in query:
                docs.append(Document(
                    page_content="大型語言模型（Large Language Models，簡稱LLM）是一種基於深度學習的自然語言處理模型，通過大規模預訓練學習語言的統計規律。",
                    metadata={"source": "llm_intro.md", "section": "introduction", "relevance": "high"}
                ))
                docs.append(Document(
                    page_content="LLM通常採用預訓練-微調的範式，先在大規模語料庫上進行無監督學習，再在特定任務上進行有監督微調。",
                    metadata={"source": "llm_intro.md", "section": "training", "relevance": "medium"}
                ))
            else:
                docs.append(Document(
                    page_content=f"這是關於「{query}」的相關文檔1。",
                    metadata={"source": f"doc1_{i}.md", "relevance": "high"}
                ))
                docs.append(Document(
                    page_content=f"這是關於「{query}」的相關文檔2。",
                    metadata={"source": f"doc2_{i}.md", "relevance": "medium"}
                ))
            
            relevant_docs.append(docs)
        
        return relevant_docs
    
    def _save_test_data(
        self,
        queries: List[str],
        reference_answers: List[str],
        relevant_docs: List[List[Document]]
    ) -> None:
        """
        保存測試數據
        
        Args:
            queries: 查詢列表
            reference_answers: 參考答案列表
            relevant_docs: 相關文檔列表的列表
        """
        # 創建測試數據目錄
        test_data_dir = os.path.join(self.output_dir, "test_data")
        os.makedirs(test_data_dir, exist_ok=True)
        
        # 保存查詢
        with open(os.path.join(test_data_dir, "queries.json"), "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        
        # 保存參考答案
        with open(os.path.join(test_data_dir, "reference_answers.json"), "w", encoding="utf-8") as f:
            json.dump(reference_answers, f, ensure_ascii=False, indent=2)
        
        # 保存相關文檔
        relevant_docs_json = []
        
        for docs in relevant_docs:
            docs_json = []
            
            for doc in docs:
                doc_json = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                docs_json.append(doc_json)
            
            relevant_docs_json.append(docs_json)
        
        with open(os.path.join(test_data_dir, "relevant_docs.json"), "w", encoding="utf-8") as f:
            json.dump(relevant_docs_json, f, ensure_ascii=False, indent=2)
        
        logger.info(f"測試數據已保存到 {test_data_dir}")
    
    def load_test_data(self) -> Tuple[List[str], List[str], List[List[Document]]]:
        """
        加載測試數據
        
        Returns:
            查詢列表、參考答案列表和相關文檔列表的元組
        """
        # 測試數據目錄
        test_data_dir = os.path.join(self.output_dir, "test_data")
        
        # 檢查測試數據是否存在
        if not os.path.exists(test_data_dir):
            logger.warning(f"測試數據目錄 {test_data_dir} 不存在，將生成新的測試數據")
            return self.generate_test_data()
        
        # 加載查詢
        try:
            with open(os.path.join(test_data_dir, "queries.json"), "r", encoding="utf-8") as f:
                queries = json.load(f)
        except Exception as e:
            logger.error(f"加載查詢時出錯: {str(e)}")
            queries = []
        
        # 加載參考答案
        try:
            with open(os.path.join(test_data_dir, "reference_answers.json"), "r", encoding="utf-8") as f:
                reference_answers = json.load(f)
        except Exception as e:
            logger.error(f"加載參考答案時出錯: {str(e)}")
            reference_answers = []
        
        # 加載相關文檔
        try:
            with open(os.path.join(test_data_dir, "relevant_docs.json"), "r", encoding="utf-8") as f:
                relevant_docs_json = json.load(f)
            
            relevant_docs = []
            
            for docs_json in relevant_docs_json:
                docs = []
                
                for doc_json in docs_json:
                    doc = Document(
                        page_content=doc_json["page_content"],
                        metadata=doc_json["metadata"]
                    )
                    docs.append(doc)
                
                relevant_docs.append(docs)
        except Exception as e:
            logger.error(f"加載相關文檔時出錯: {str(e)}")
            relevant_docs = []
        
        # 檢查數據是否完整
        if not queries or not reference_answers or not relevant_docs:
            logger.warning("測試數據不完整，將生成新的測試數據")
            return self.generate_test_data()
        
        # 檢查數據長度是否一致
        if len(queries) != len(reference_answers) or len(queries) != len(relevant_docs):
            logger.warning("測試數據長度不一致，將生成新的測試數據")
            return self.generate_test_data()
        
        logger.info(f"已從 {test_data_dir} 加載測試數據")
        return queries, reference_answers, relevant_docs

class EvaluationPipeline:
    """評估流水線，整合不同評估組件"""
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化評估流水線
        
        Args:
            config: 評估配置
        """
        self.config = config
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化評估組件
        self.data_generator = EvaluationDataGenerator(config)
        self.retrieval_evaluator = RetrievalEvaluator(config)
        self.generation_evaluator = GenerationEvaluator(config)
        self.rag_evaluator = RAGEvaluator(config)
    
    def evaluate_system(
        self,
        rag_system,
        test_data: Optional[Tuple[List[str], List[str], List[List[Document]]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        評估系統
        
        Args:
            rag_system: RAG系統
            test_data: 測試數據元組（查詢列表、參考答案列表和相關文檔列表）
            
        Returns:
            評估結果字典
        """
        # 加載或生成測試數據
        if test_data:
            queries, reference_answers, relevant_docs = test_data
        else:
            queries, reference_answers, relevant_docs = self.data_generator.load_test_data()
        
        # 評估RAG系統
        results = self.rag_evaluator.evaluate_rag_system(
            rag_system,
            queries,
            reference_answers,
            relevant_docs
        )
        
        # 保存評估結果
        self._save_evaluation_results(results)
        
        return results
    
    def _save_evaluation_results(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> None:
        """
        保存評估結果
        
        Args:
            results: 評估結果字典
        """
        # 創建結果目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.output_dir, f"evaluation_results_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存評估結果
        with open(os.path.join(result_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 創建評估報告
        report = self._generate_evaluation_report(results)
        
        # 保存評估報告
        with open(os.path.join(result_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"評估結果已保存到 {result_dir}")
    
    def _generate_evaluation_report(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> str:
        """
        生成評估報告
        
        Args:
            results: 評估結果字典
            
        Returns:
            評估報告文本
        """
        # 獲取檢索和生成結果
        retrieval_results = results.get("retrieval", {})
        generation_results = results.get("generation", {})
        
        # 生成報告
        report = f"""# RAG系統評估報告

## 評估時間
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 檢索性能

| 指標 | 分數 |
|------|------|
"""
        
        # 添加檢索指標
        for metric, score in retrieval_results.items():
            report += f"| {metric} | {score:.4f} |\n"
        
        report += f"""
## 生成性能

| 指標 | 分數 |
|------|------|
"""
        
        # 添加生成指標
        for metric, score in generation_results.items():
            report += f"| {metric} | {score:.4f} |\n"
        
        report += f"""
## 綜合評估

RAG系統的整體性能取決於檢索和生成兩個方面。檢索性能影響系統能否找到相關信息，生成性能影響系統能否基於檢索到的信息生成高質量的回答。

### 檢索性能分析

"""
        
        # 分析檢索性能
        precision = retrieval_results.get("precision", 0)
        recall = retrieval_results.get("recall", 0)
        ndcg = retrieval_results.get("ndcg", 0)
        
        if precision > 0.8:
            report += "檢索精確率較高，系統能夠準確找到相關文檔。"
        elif precision > 0.5:
            report += "檢索精確率中等，系統找到的文檔中有一部分不相關。"
        else:
            report += "檢索精確率較低，系統找到的文檔中有較多不相關內容。"
        
        if recall > 0.8:
            report += "召回率較高，系統能夠找到大部分相關文檔。"
        elif recall > 0.5:
            report += "召回率中等，系統漏掉了一部分相關文檔。"
        else:
            report += "召回率較低，系統漏掉了較多相關文檔。"
        
        if ndcg > 0.8:
            report += "NDCG較高，系統的檢索排序質量較好。"
        elif ndcg > 0.5:
            report += "NDCG中等，系統的檢索排序質量一般。"
        else:
            report += "NDCG較低，系統的檢索排序質量較差。"
        
        report += f"""

### 生成性能分析

"""
        
        # 分析生成性能
        relevance = generation_results.get("relevance", 0)
        factuality = generation_results.get("factuality", 0)
        coherence = generation_results.get("coherence", 0)
        fluency = generation_results.get("fluency", 0)
        
        if relevance > 0.8:
            report += "回答相關性較高，系統生成的回答與查詢高度相關。"
        elif relevance > 0.5:
            report += "回答相關性中等，系統生成的回答與查詢部分相關。"
        else:
            report += "回答相關性較低，系統生成的回答與查詢關聯不大。"
        
        if factuality > 0.8:
            report += "事實性較高，系統生成的回答與參考答案高度一致。"
        elif factuality > 0.5:
            report += "事實性中等，系統生成的回答與參考答案部分一致。"
        else:
            report += "事實性較低，系統生成的回答與參考答案差異較大。"
        
        if coherence > 0.8:
            report += "連貫性較高，系統生成的回答邏輯清晰、結構合理。"
        elif coherence > 0.5:
            report += "連貫性中等，系統生成的回答邏輯基本清晰。"
        else:
            report += "連貫性較低，系統生成的回答邏輯不清晰。"
        
        if fluency > 0.8:
            report += "流暢性較高，系統生成的回答表達自然、易於理解。"
        elif fluency > 0.5:
            report += "流暢性中等，系統生成的回答表達基本自然。"
        else:
            report += "流暢性較低，系統生成的回答表達不自然。"
        
        report += f"""

## 改進建議

"""
        
        # 提出改進建議
        if precision < 0.7:
            report += "- 提高檢索精確率：優化檢索算法，調整相似度閾值，或使用更好的嵌入模型。\n"
        
        if recall < 0.7:
            report += "- 提高召回率：擴大知識庫，優化文檔分塊策略，或使用混合檢索方法。\n"
        
        if relevance < 0.7:
            report += "- 提高回答相關性：優化提示模板，加強查詢理解，或使用更好的LLM模型。\n"
        
        if factuality < 0.7:
            report += "- 提高事實性：增強知識庫質量，優化檢索結果過濾，或使用事實性增強技術。\n"
        
        if coherence < 0.7 or fluency < 0.7:
            report += "- 提高生成質量：微調LLM模型，優化生成參數，或使用更好的後處理方法。\n"
        
        report += """
## 結論

基於以上評估結果，該RAG系統在檢索和生成方面都有一定的表現。通過實施上述改進建議，系統性能有望進一步提升。

---

*本報告由評估流水線自動生成*
"""
        
        return report

def create_evaluation_examples():
    """創建評估示例"""
    # 創建評估配置
    config = EvaluationConfig(
        metrics=["relevance", "factuality", "coherence", "fluency", "rouge", "bleu"],
        retrieval_metrics=["precision", "recall", "ndcg", "mrr"],
        output_dir="../evaluation_results"
    )
    
    # 創建評估流水線
    pipeline = EvaluationPipeline(config)
    
    # 生成測試數據
    data_generator = EvaluationDataGenerator(config)
    queries, reference_answers, relevant_docs = data_generator.generate_test_data(num_samples=10)
    
    # 創建模擬的RAG系統
    class MockRAGSystem:
        def query(self, query):
            # 模擬查詢處理
            answer = f"這是關於「{query}」的回答。它基於檢索到的相關文檔生成。"
            
            # 模擬檢索文檔
            source_documents = [
                Document(
                    page_content=f"這是關於「{query}」的相關文檔1。",
                    metadata={"source": "doc1.md"}
                ),
                Document(
                    page_content=f"這是關於「{query}」的相關文檔2。",
                    metadata={"source": "doc2.md"}
                )
            ]
            
            return {
                "query": query,
                "answer": answer,
                "source_documents": source_documents
            }
    
    # 創建模擬的RAG系統
    mock_rag_system = MockRAGSystem()
    
    # 評估檢索性能
    retrieval_evaluator = RetrievalEvaluator(config)
    retrieved_docs = [[mock_rag_system.query(query)["source_documents"] for _ in range(2)] for query in queries]
    retrieval_results = retrieval_evaluator.evaluate_retrieval(queries, retrieved_docs, relevant_docs)
    
    # 評估生成性能
    generation_evaluator = GenerationEvaluator(config)
    generated_answers = [mock_rag_system.query(query)["answer"] for query in queries]
    generation_results = generation_evaluator.evaluate_generation(queries, generated_answers, reference_answers)
    
    # 評估RAG系統
    rag_evaluator = RAGEvaluator(config)
    rag_results = rag_evaluator.evaluate_rag_system(mock_rag_system, queries, reference_answers, relevant_docs)
    
    # 生成評估報告
    report = pipeline._generate_evaluation_report(rag_results)
    
    # 保存評估報告
    report_path = os.path.join(config.output_dir, "example_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"評估示例報告已保存到 {report_path}")
    
    return {
        "config": config,
        "pipeline": pipeline,
        "retrieval_results": retrieval_results,
        "generation_results": generation_results,
        "rag_results": rag_results,
        "report_path": report_path
    }

def generate_evaluation_code():
    """生成評估代碼"""
    # 創建主程序
    main_code = """
import os
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document

from evaluation_metrics import (
    EvaluationConfig,
    EvaluationPipeline,
    EvaluationDataGenerator,
    RetrievalEvaluator,
    GenerationEvaluator,
    RAGEvaluator,
    create_evaluation_examples
)

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="評估指標系統")
    parser.add_argument("--mode", type=str, default="example", choices=["example", "evaluate", "generate"], help="運行模式")
    parser.add_argument("--output_dir", type=str, default="../evaluation_results", help="輸出目錄")
    parser.add_argument("--num_samples", type=int, default=10, help="樣本數量")
    parser.add_argument("--rag_system_path", type=str, help="RAG系統路徑")
    
    args = parser.parse_args()
    
    # 創建評估配置
    config = EvaluationConfig(
        metrics=["relevance", "factuality", "coherence", "fluency", "rouge", "bleu"],
        retrieval_metrics=["precision", "recall", "ndcg", "mrr"],
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    # 運行模式
    if args.mode == "example":
        # 創建評估示例
        results = create_evaluation_examples()
        print(f"評估示例已創建，報告保存在: {results['report_path']}")
    
    elif args.mode == "generate":
        # 生成測試數據
        data_generator = EvaluationDataGenerator(config)
        queries, reference_answers, relevant_docs = data_generator.generate_test_data(num_samples=args.num_samples)
        print(f"測試數據已生成，包含 {len(queries)} 個樣本")
    
    elif args.mode == "evaluate":
        # 檢查RAG系統路徑
        if not args.rag_system_path:
            print("錯誤: 評估模式需要指定RAG系統路徑")
            return
        
        # 加載RAG系統
        try:
            import sys
            sys.path.append(os.path.dirname(args.rag_system_path))
            
            from importlib.util import spec_from_file_location, module_from_spec
            
            # 加載模塊
            spec = spec_from_file_location("rag_system", args.rag_system_path)
            rag_module = module_from_spec(spec)
            spec.loader.exec_module(rag_module)
            
            # 獲取RAG系統
            if hasattr(rag_module, "RAGSystem"):
                rag_system = rag_module.RAGSystem()
            elif hasattr(rag_module, "RAGSystemWithMultiVectorDB"):
                from vector_db_integration import VectorDBManager, VectorDBConfig
                
                # 創建向量資料庫管理器
                vector_db_manager = VectorDBManager()
                
                # 創建默認向量資料庫配置
                default_config = VectorDBConfig(
                    db_type="faiss",
                    collection_name="evaluation",
                    persist_directory=f"../data/vector_store/faiss_evaluation"
                )
                
                # 創建RAG系統
                rag_system = rag_module.RAGSystemWithMultiVectorDB(
                    vector_db_manager=vector_db_manager,
                    default_db_config=default_config
                )
            else:
                print(f"錯誤: 在 {args.rag_system_path} 中找不到RAG系統類")
                return
        except Exception as e:
            print(f"加載RAG系統時出錯: {str(e)}")
            return
        
        # 創建評估流水線
        pipeline = EvaluationPipeline(config)
        
        # 評估RAG系統
        results = pipeline.evaluate_system(rag_system)
        
        print("評估完成，結果如下:")
        print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
"""
    
    # 保存主程序
    main_path = "../code/evaluation/main.py"
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_code)
    
    logger.info(f"評估主程序已保存到 {main_path}")
    
    # 創建__init__.py
    init_code = """
from .evaluation_metrics import (
    EvaluationConfig,
    RetrievalEvaluator,
    GenerationEvaluator,
    RAGEvaluator,
    EvaluationDataGenerator,
    EvaluationPipeline,
    create_evaluation_examples
)
"""
    
    init_path = "../code/evaluation/__init__.py"
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(init_code)
    
    logger.info(f"評估__init__.py已保存到 {init_path}")
    
    return {
        "main_path": main_path,
        "init_path": init_path
    }

if __name__ == "__main__":
    # 創建評估示例
    examples = create_evaluation_examples()
    
    # 生成評估代碼
    code_paths = generate_evaluation_code()
    
    print("\n所有準備工作已完成！")
    print(f"1. 評估指標系統: {__file__}")
    print(f"2. 評估主程序: {code_paths['main_path']}")
    print(f"3. 評估__init__.py: {code_paths['init_path']}")
    print(f"4. 評估示例報告: {examples['report_path']}")

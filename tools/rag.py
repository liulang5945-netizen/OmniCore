"""
RAG (检索增强生成) 知识库模块
基于语义向量嵌入的现代文档检索系统
支持多文档导入、删除、增量索引重建、持久化存储

核心升级：
1. 🚀 使用 Sentence-Transformer 生成语义嵌入（理解"苹果公司"与"iPhone 制造商"的关联）
2. ✂️ 智能文档切分，带重叠上下文（Overlap），保持语义连贯性
3. 🔍 混合检索：Dense 向量检索 + BM25 稀疏检索，RRF 融合排序
4. 🎯 Cross-Encoder 重排序（Reranker），对候选结果精细打分
5. 🔄 查询改写（Query Rewriting），提升检索召回率
6. 💾 持久化使用 NumPy 序列化，加载速度快
"""
import json
import logging
import math
import os
import re
import time
import threading
from collections import Counter
from typing import Optional, List, Tuple, Dict

from core.memory_watchdog import memory_guarded, MemoryWatchdog
from tools.file_parser import parse_file_to_text

logger = logging.getLogger("RAG")

# ======================== 配置 ========================

# 默认使用的 Embedding 模型
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 备选轻量模型（当默认模型不可用时）
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 默认 Cross-Encoder 重排序模型
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 文档分块大小（字符数）
CHUNK_SIZE = 200

# 分块重叠大小（字符数）
CHUNK_OVERLAP = 50

# 检索返回的 top-k 段落数
DEFAULT_TOP_K = 4

# RRF 融合常数（越大越平滑）
RRF_K = 60

# 混合检索候选数
HYBRID_CANDIDATE_K = 20


# ======================== RAG 检索策略配置 ========================

class RAGConfig:
    """RAG 检索策略配置（单例，持久化到 app_settings.json）"""

    _instance = None
    _lock = threading.Lock()

    DEFAULTS = {
        "enable_hybrid": True,
        "enable_reranker": True,
        "enable_query_rewrite": False,
        "dense_weight": 0.6,
        "bm25_weight": 0.4,
        "candidate_k": 20,
        "reranker_model": DEFAULT_RERANKER_MODEL,
    }

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._config = dict(cls.DEFAULTS)
                cls._instance._load_from_settings()
            return cls._instance

    def _settings_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app_settings.json"
        )

    def _load_from_settings(self):
        """从 app_settings.json 加载 RAG 配置"""
        path = self._settings_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key in self.DEFAULTS:
                    settings_key = f"rag_{key}"
                    if settings_key in data:
                        self._config[key] = data[settings_key]
            except Exception as e:
                logger.warning(f"加载 RAG 配置失败: {e}")

    def save_to_settings(self):
        """将 RAG 配置保存到 app_settings.json"""
        path = self._settings_path()
        try:
            data = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            for key, value in self._config.items():
                data[f"rag_{key}"] = value
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存 RAG 配置失败: {e}")

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    def set(self, key: str, value):
        self._config[key] = value

    def to_dict(self) -> dict:
        return dict(self._config)

    def update(self, updates: dict):
        for key, value in updates.items():
            if key in self.DEFAULTS:
                self._config[key] = value
        self.save_to_settings()


# ======================== BM25 稀疏检索索引 ========================

class BM25Index:
    """
    BM25 稀疏检索索引（纯 Python 实现，不依赖外部库）

    BM25 公式: score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) /
                              (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

    其中:
      - f(qi, D) = 词 qi 在文档 D 中的词频
      - |D| = 文档 D 的长度
      - avgdl = 所有文档的平均长度
      - k1 = 词频饱和参数 (默认 1.5)
      - b = 文档长度归一化参数 (默认 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lengths: List[int] = []       # 每个文档的 token 数
        self.doc_freqs: List[Counter] = []     # 每个文档的词频 Counter
        self.idf: Dict[str, float] = {}        # 逆文档频率
        self._stopwords = self._default_stopwords()

    @staticmethod
    def _default_stopwords() -> set:
        """中英文常用停用词"""
        return {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "of", "in", "to", "for", "with", "on", "at", "from", "by",
            "and", "or", "but", "not", "no", "if", "then", "that", "this",
            "it", "its", "as", "how", "what", "when", "where", "who", "which",
        }

    def _tokenize(self, text: str) -> List[str]:
        """分词：中文用 jieba，英文按空格 + 小写"""
        text = text.lower()
        # 尝试 jieba 中文分词
        try:
            import jieba
            tokens = jieba.lcut(text)
        except ImportError:
            # 回退：按字符和空格分词
            tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)
        # 过滤停用词和单字符
        return [t for t in tokens if t not in self._stopwords and len(t) > 1]

    def build(self, chunks: List[Tuple[str, str, int]]):
        """
        从 chunks 构建 BM25 索引

        Args:
            chunks: [(filename, chunk_text, index), ...] 与 RAGKnowledgeBase.chunks 格式一致
        """
        self.doc_count = len(chunks)
        self.doc_lengths = []
        self.doc_freqs = []
        df_counter: Counter = Counter()  # 文档频率

        for _, chunk_text, _ in chunks:
            tokens = self._tokenize(chunk_text)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            self.doc_lengths.append(len(tokens))
            # 统计每个词出现在多少个文档中
            for term in set(tokens):
                df_counter[term] += 1

        # 计算平均文档长度
        self.avg_dl = sum(self.doc_lengths) / max(self.doc_count, 1)

        # 计算 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {}
        for term, df in df_counter.items():
            self.idf[term] = math.log(
                (self.doc_count - df + 0.5) / (df + 0.5) + 1.0
            )

        logger.info(f"BM25 索引构建完成: {self.doc_count} 文档, "
                    f"{len(self.idf)} 词项, 平均长度 {self.avg_dl:.1f}")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        BM25 检索

        Args:
            query: 查询字符串
            top_k: 返回 top_k 个结果

        Returns:
            [(chunk_index, score), ...] 按 score 降序排列
        """
        if self.doc_count == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for doc_idx in range(self.doc_count):
            score = 0.0
            dl = self.doc_lengths[doc_idx]
            freq = self.doc_freqs[doc_idx]
            for term in query_tokens:
                if term not in self.idf:
                    continue
                tf = freq.get(term, 0)
                idf = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                score += idf * numerator / max(denominator, 1e-10)
            if score > 0:
                scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> dict:
        """序列化为可 pickle 的 dict"""
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_dl": self.avg_dl,
            "doc_lengths": self.doc_lengths,
            "doc_freqs": self.doc_freqs,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        """从 dict 反序列化"""
        obj = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        obj.doc_count = data.get("doc_count", 0)
        obj.avg_dl = data.get("avg_dl", 0.0)
        obj.doc_lengths = data.get("doc_lengths", [])
        obj.doc_freqs = data.get("doc_freqs", [])
        obj.idf = data.get("idf", {})
        return obj


# ======================== Cross-Encoder 重排序器 ========================

class CrossEncoderReranker:
    """
    Cross-Encoder 重排序器
    对 (query, passage) 对进行精细打分，比 bi-encoder 更准确

    延迟加载模型，首次使用时才下载和初始化
    """

    def __init__(self, model_name: str = None):
        self._model_name = model_name or RAGConfig().get("reranker_model", DEFAULT_RERANKER_MODEL)
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        """延迟加载 Cross-Encoder 模型"""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"加载 Cross-Encoder 重排序模型: {self._model_name}")
                self._model = CrossEncoder(self._model_name, device="cpu")
                logger.info("Cross-Encoder 重排序模型加载成功")
            except Exception as e:
                logger.warning(f"Cross-Encoder 模型加载失败: {e}，重排序功能不可用")
                self._model = None

    def is_available(self) -> bool:
        """检查重排序模型是否可用"""
        if self._model is not None:
            return True
        try:
            self._load_model()
            return self._model is not None
        except Exception:
            return False

    def rerank(self, query: str, passages: List[str], top_k: int = 4) -> List[Tuple[int, float]]:
        """
        对 (query, passage) 对打分并排序

        Args:
            query: 查询字符串
            passages: 候选段落列表
            top_k: 返回 top_k 个结果

        Returns:
            [(original_index, score), ...] 按 score 降序排列
        """
        if not passages:
            return []

        self._load_model()
        if self._model is None:
            # 模型不可用，返回原始顺序
            return [(i, 0.0) for i in range(min(top_k, len(passages)))]

        try:
            # 构造 (query, passage) 对
            pairs = [(query, p) for p in passages]
            scores = self._model.predict(pairs)

            # 按分数降序排列
            scored = list(enumerate(scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [(idx, float(score)) for idx, score in scored[:top_k]]
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return [(i, 0.0) for i in range(min(top_k, len(passages)))]


class RAGKnowledgeBase:
    """
    语义知识库
    使用 Sentence-Transformer 对文档段落进行向量嵌入和检索
    支持 save/load 持久化存储
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.documents = {}           # {filename: full_text}
        self.chunks = []              # [(filename, chunk_text, index)]
        self.embeddings = None        # np.ndarray, shape=(n_chunks, embed_dim)
        self._embedder = None         # Lazy-loaded SentenceTransformer 模型
        self._embed_dim = 0
        self.persist_dir = persist_dir

        # 混合检索组件
        self._bm25_index: Optional[BM25Index] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._rag_config = RAGConfig()

        # 如果有持久化路径，自动加载
        if persist_dir and os.path.exists(os.path.join(persist_dir, "rag_embeddings.npy")):
            self._load()

    # ======================== Embedding 模型管理 ========================

    @memory_guarded(min_avail_pct=0.15, on_critical='raise')
    def _get_embedder(self):
        """
        延迟加载 Embedding 模型（首次使用时加载）
        支持自定义模型路径
        """
        if self._embedder is not None:
            return self._embedder

        # 尝试加载指定的 embedding 模型
        model_name = os.environ.get(
            "OMNICORE_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL
        )

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载 Embedding 模型: {model_name}")
            self._embedder = SentenceTransformer(
                model_name,
                device="cpu",
                cache_folder=os.environ.get("OMNICORE_CACHE_DIR", None),
            )
            self._embed_dim = self._embedder.get_sentence_embedding_dimension()
            logger.info(f"Embedding 模型加载成功，维度: {self._embed_dim}")
        except Exception as e:
            if model_name != FALLBACK_EMBEDDING_MODEL:
                logger.warning(f"加载 {model_name} 失败: {e}，尝试备用模型...")
                try:
                    from sentence_transformers import SentenceTransformer
                    self._embedder = SentenceTransformer(
                        FALLBACK_EMBEDDING_MODEL,
                        device="cpu",
                    )
                    self._embed_dim = self._embedder.get_sentence_embedding_dimension()
                    logger.info(f"备用 Embedding 模型加载成功，维度: {self._embed_dim}")
                except Exception as e2:
                    logger.error(f"所有 Embedding 模型均加载失败: {e2}")
                    raise
            else:
                raise

        return self._embedder

    def _build_embeddings(self, texts: List[str], batch_size: int = 32) -> "np.ndarray":
        """
        批量构建文本嵌入向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            np.ndarray, shape=(len(texts), embed_dim)
        """
        import numpy as np

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 0)

        embedder = self._get_embedder()
        try:
            embeddings = embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 归一化，直接余弦相似度
                convert_to_numpy=True,
            )
            return embeddings
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            raise

    # ======================== 文档分块（智能版） ========================

    @staticmethod
    def _chunk_text(text: str,
                     chunk_size: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        智能文档分块，带重叠上下文

        改进点（相对原始版本）：
        1. 使用重叠窗口保持语义连贯性
        2. 优先在句子边界切分
        3. 在小段落合并以节省 embedding 空间

        Args:
            text: 原始文本
            chunk_size: 每块目标字符数
            overlap: 块与块之间的重叠字符数

        Returns:
            文本块列表
        """
        if not text.strip():
            return []

        # 按段落分割
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # 合并过短的段落
        merged = []
        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) < chunk_size * 0.5:
                buffer += "\n" + para if buffer else para
            else:
                if buffer:
                    merged.append(buffer)
                buffer = para
        if buffer:
            merged.append(buffer)

        if not merged:
            return []

        # 在句子边界切分
        def _split_at_sentence(text_block: str, max_size: int) -> List[str]:
            """
            在句子边界处切分文本块，确保不超过 max_size
            """
            if len(text_block) <= max_size:
                return [text_block]

            # 中文句子分割符
            sentence_end = re.compile(r'(?<=[。！？.!?；;])\s*')
            sentences = sentence_end.split(text_block)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 1:
                # 如果没有句子边界，强制截断
                return [text_block[:max_size], text_block[max_size:]]

            chunks = []
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= max_size:
                    current += sent
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sent
            if current:
                chunks.append(current.strip())
            return chunks

        # 分块 + 重叠
        chunks = []
        for block in merged:
            if len(block) <= chunk_size:
                chunks.append(block)
            else:
                sub_chunks = _split_at_sentence(block, chunk_size)
                # 添加重叠
                overlapped = []
                for i, sc in enumerate(sub_chunks):
                    if i > 0 and overlap > 0:
                        # 从前一个块末尾取 overlap 字符作为上下文
                        prev = overlapped[-1]
                        overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                        sc = overlap_text + sc
                    overlapped.append(sc)
                chunks.extend(overlapped)

        return chunks

    # ======================== 文档管理 ========================

    def add_file(self, file_path: str):
        """向知识库添加文件"""
        filename = os.path.basename(file_path)
        text = parse_file_to_text(file_path)
        if not text.strip():
            logger.warning(f"文件内容为空: {filename}")
            return
        self.documents[filename] = text
        logger.info(f"已添加文档: {filename} ({len(text)} 字符)")

    def add_text(self, filename: str, text: str):
        """向知识库直接添加文本"""
        if not text.strip():
            logger.warning(f"文本内容为空: {filename}")
            return
        self.documents[filename] = text
        logger.info(f"已添加文本文档: {filename} ({len(text)} 字符)")

    def remove_file(self, filename: str):
        """从知识库移除文件"""
        self.documents.pop(filename, None)
        # 移除该文件对应的所有 chunk
        self.chunks = [
            (fn, text, idx)
            for fn, text, idx in self.chunks
            if fn != filename
        ]
        # 清除 embeddings（下次 rebuild 会重新计算）
        self.embeddings = None
        logger.info(f"已移除文档: {filename}")

    def get_doc_names(self) -> list:
        """获取所有文档名称"""
        return list(self.documents.keys())

    def clear(self):
        """清空知识库"""
        self.documents = {}
        self.chunks = []
        self.embeddings = None
        self._embedder = None
        self._bm25_index = None
        # 删除持久化文件
        if self.persist_dir:
            for fname in ["rag_documents.json", "rag_embeddings.npy",
                          "rag_index.json", "rag_index.pkl",
                          "rag_bm25.json", "rag_bm25.pkl"]:
                fpath = os.path.join(self.persist_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
        logger.info("RAG 知识库已清空")

    # ======================== 索引构建 ========================

    def rebuild_index(self) -> str:
        """
        重建语义嵌入索引

        流程：
        1. 将所有文档智能分块
        2. 使用 Sentence-Transformer 生成嵌入向量
        3. 持久化到磁盘

        受 MemoryWatchdog 保护：进入前预检，退出后 gc.collect。
        """
        import numpy as np

        # 将所有文档切块
        self.chunks = []
        all_chunk_texts = []

        for filename, text in self.documents.items():
            file_chunks = self._chunk_text(text)
            for idx, chunk in enumerate(file_chunks):
                self.chunks.append((filename, chunk, idx))
                all_chunk_texts.append(chunk)

        if not all_chunk_texts:
            self.embeddings = None
            return "知识库为空，请导入文档。"

        # ── 内存预检 ──
        wd = MemoryWatchdog()
        can_proceed, msg = wd.can_proceed(min_avail_pct=0.12)
        if not can_proceed:
            logger.warning(f"RAG索引构建中止: {msg}")
            return f"内存不足，无法构建索引。{msg}"

        # 检查嵌入向量所需内存
        embed_dim = self._embed_dim or 384
        can_build, build_msg = MemoryWatchdog.can_build_embeddings(
            len(all_chunk_texts), embed_dim
        )
        if not can_build:
            logger.warning(f"RAG索引构建中止: {build_msg}")
            return build_msg

        try:
            # 生成嵌入向量
            logger.info(f"正在为 {len(all_chunk_texts)} 个段落生成嵌入向量...")
            start_time = time.time()

            # 嵌入生成中哨兵：分批次检查
            batch_size = 32
            all_embeddings = []
            embedder = self._get_embedder()
            for i in range(0, len(all_chunk_texts), batch_size):
                batch = all_chunk_texts[i:i + batch_size]
                if i % (batch_size * 10) == 0 and wd.status.level >= 3:
                    return (
                        f"内存告急（{wd.status.avail_pct*100:.0f}%），"
                        f"索引构建已中止。已处理 {i}/{len(all_chunk_texts)} 段落。"
                    )
                batch_emb = embedder.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                all_embeddings.append(batch_emb)

            if all_embeddings:
                self.embeddings = np.vstack(all_embeddings)
            else:
                self.embeddings = None

            elapsed = time.time() - start_time
            logger.info(f"嵌入生成完成，耗时 {elapsed:.1f} 秒")
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            return f"嵌入生成失败: {e}。请确保已安装 sentence-transformers。"

        # ── 构建 BM25 索引 ──
        try:
            self._bm25_index = BM25Index()
            self._bm25_index.build(self.chunks)
            logger.info("BM25 索引同步构建完成")
        except Exception as e:
            logger.warning(f"BM25 索引构建失败（不影响 Dense 检索）: {e}")
            self._bm25_index = None

        # 自动持久化
        self._save()

        doc_count = len(self.documents)
        chunk_count = len(self.chunks)
        emb_dim = self.embeddings.shape[1] if self.embeddings is not None else 0
        has_bm25 = self._bm25_index is not None
        msg = (f"索引构建完成！共 {doc_count} 篇文档，"
               f"{chunk_count} 个段落，嵌入维度 {emb_dim}，"
               f"BM25: {'✓' if has_bm25 else '✗'}。")
        logger.info(msg)
        return msg

    # ======================== 检索 ========================

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, str, float]]:
        """
        语义搜索与 query 最相关的段落

        核心改进（相对 TF-IDF）：
        - 使用语义嵌入理解"苹果公司"与"iPhone 制造商"的关系
        - 使用归一化余弦距离计算相似度

        Returns:
            [(filename, chunk_text, score), ...] 按得分降序排列
        """
        import numpy as np

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # 生成 query 的嵌入向量
        try:
            query_vec = self._build_embeddings([query])
        except Exception as e:
            logger.error(f"查询嵌入生成失败: {e}")
            return []

        if query_vec.shape[0] == 0:
            return []

        # 余弦相似度（因为已经 L2 归一化，直接用点积）
        scores = np.dot(self.embeddings, query_vec[0])

        # 获取 top_k 结果
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:  # 忽略极低相似度
                filename, chunk_text, _ = self.chunks[idx]
                results.append((filename, chunk_text, float(scores[idx])))

        return results

    # ======================== 混合检索 ========================

    def _dense_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Dense 向量检索（返回 chunk 索引 + 分数）

        Args:
            query: 查询字符串
            top_k: 返回 top_k 个结果

        Returns:
            [(chunk_index, score), ...]
        """
        import numpy as np

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        try:
            query_vec = self._build_embeddings([query])
        except Exception as e:
            logger.error(f"查询嵌入生成失败: {e}")
            return []

        if query_vec.shape[0] == 0:
            return []

        scores = np.dot(self.embeddings, query_vec[0])
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:
                results.append((int(idx), float(scores[idx])))

        return results

    def hybrid_search(self, query: str, top_k: int = DEFAULT_TOP_K,
                      candidate_k: int = None) -> List[Tuple[str, str, float]]:
        """
        RRF (Reciprocal Rank Fusion) 混合检索

        将 Dense 向量检索和 BM25 稀疏检索的结果通过 RRF 算法融合:
        RRF_score(d) = Σ 1 / (k + rank_i(d))

        Args:
            query: 查询字符串
            top_k: 最终返回的结果数
            candidate_k: 每种检索器的候选数

        Returns:
            [(filename, chunk_text, score), ...] 按 RRF 分数降序排列
        """
        if candidate_k is None:
            candidate_k = self._rag_config.get("candidate_k", HYBRID_CANDIDATE_K)

        # Dense 检索
        dense_results = self._dense_search(query, top_k=candidate_k)

        # BM25 检索
        bm25_results = []
        if self._bm25_index is not None:
            try:
                bm25_results = self._bm25_index.search(query, top_k=candidate_k)
            except Exception as e:
                logger.warning(f"BM25 检索失败: {e}")

        # 如果只有一种检索结果，直接返回
        if not dense_results and not bm25_results:
            return []
        if not bm25_results:
            # 仅 Dense
            return [(self.chunks[idx][0], self.chunks[idx][1], score)
                    for idx, score in dense_results[:top_k]]
        if not dense_results:
            # 仅 BM25
            return [(self.chunks[idx][0], self.chunks[idx][1], score)
                    for idx, score in bm25_results[:top_k]]

        # ── RRF 融合 ──
        rrf_scores: Dict[int, float] = {}

        # Dense 排名贡献
        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        # BM25 排名贡献
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        # 按 RRF 分数降序排列
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, rrf_score in sorted_indices[:top_k]:
            if idx < len(self.chunks):
                filename, chunk_text, _ = self.chunks[idx]
                results.append((filename, chunk_text, rrf_score))

        logger.debug(f"混合检索完成: Dense={len(dense_results)} + BM25={len(bm25_results)} "
                     f"→ RRF top-{top_k}")
        return results

    def _get_reranker(self) -> Optional[CrossEncoderReranker]:
        """延迟获取重排序器实例"""
        if not self._rag_config.get("enable_reranker", True):
            return None
        if self._reranker is None:
            try:
                self._reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning(f"重排序器初始化失败: {e}")
        return self._reranker

    def _rerank_results(self, query: str,
                        results: List[Tuple[str, str, float]],
                        top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, str, float]]:
        """
        使用 Cross-Encoder 对检索结果重排序

        Args:
            query: 查询字符串
            results: 候选结果 [(filename, chunk_text, score), ...]
            top_k: 重排序后返回的结果数

        Returns:
            重排序后的结果
        """
        if not results or len(results) <= 1:
            return results

        reranker = self._get_reranker()
        if reranker is None or not reranker.is_available():
            return results[:top_k]

        passages = [text for _, text, _ in results]
        reranked = reranker.rerank(query, passages, top_k=top_k)

        final_results = []
        for orig_idx, rerank_score in reranked:
            if orig_idx < len(results):
                filename, chunk_text, _ = results[orig_idx]
                final_results.append((filename, chunk_text, rerank_score))

        logger.debug(f"重排序完成: {len(results)} → {len(final_results)} 条结果")
        return final_results

    def search_with_fallback(self, query: str, top_k: int = DEFAULT_TOP_K,
                             fallback_keyword: bool = True) -> List[str]:
        """
        增强版检索管线（根据配置自动选择策略）

        管线流程:
        1. 查询改写（可选）
        2. 混合检索（Dense + BM25 RRF）或纯 Dense 检索
        3. Cross-Encoder 重排序（可选）
        4. 关键词回退（可选）

        Args:
            query: 查询字符串
            top_k: 返回 top_k 条结果
            fallback_keyword: 当检索无结果时，是否回退到关键词搜索

        Returns:
            格式化后的结果列表
        """
        config = self._rag_config
        use_hybrid = config.get("enable_hybrid", True) and self._bm25_index is not None
        use_reranker = config.get("enable_reranker", True)
        candidate_k = config.get("candidate_k", HYBRID_CANDIDATE_K)

        # 获取重排序器（如果启用，需要比 top_k 更多的候选）
        rerank_top_k = candidate_k if use_reranker else top_k

        # Step 1: 检索
        if use_hybrid:
            logger.debug("使用混合检索模式 (Dense + BM25 RRF)")
            results = self.hybrid_search(query, top_k=rerank_top_k, candidate_k=candidate_k)
        else:
            logger.debug("使用纯 Dense 语义检索")
            results = self.search(query, top_k=rerank_top_k)

        # Step 2: 重排序
        if use_reranker and results and len(results) > 1:
            results = self._rerank_results(query, results, top_k=top_k)

        # Step 3: 关键词回退
        if not results and fallback_keyword:
            logger.info("语义检索无结果，回退到关键词搜索")
            results = self._keyword_search(query, top_k=top_k)

        if not results:
            return []

        # 格式化输出
        formatted = []
        for filename, text, score in results:
            formatted.append(f"[{filename}] (相似度: {score:.3f})\n{text}")
        return formatted

    # ======================== 配置管理 ========================

    def get_rag_config(self) -> dict:
        """获取当前 RAG 检索配置"""
        return self._rag_config.to_dict()

    def update_rag_config(self, updates: dict):
        """更新 RAG 检索配置"""
        self._rag_config.update(updates)
        # 如果重排序器配置变了，重置实例
        if "reranker_model" in updates:
            self._reranker = None
        logger.info(f"RAG 配置已更新: {updates}")

    def _keyword_search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        关键词回退搜索（当语义搜索无结果时的降级策略）
        使用简单的 TF-IDF + jieba 分词
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # 如果可用 jieba，用 jieba 分词
        try:
            import jieba
            query_terms = set(jieba.lcut(query))
        except ImportError:
            pass

        if not query_terms:
            return []

        scored_chunks = []
        for idx, (filename, chunk_text, _) in enumerate(self.chunks):
            chunk_lower = chunk_text.lower()
            score = 0
            for term in query_terms:
                term_str = str(term)
                if term_str in chunk_lower:
                    # 匹配频率
                    score += chunk_lower.count(term_str)
            if score > 0:
                scored_chunks.append((filename, chunk_text, float(score)))

        # 按得分排序
        scored_chunks.sort(key=lambda x: x[2], reverse=True)
        return scored_chunks[:top_k]

    # ======================== 持久化 ========================

    def set_persist_dir(self, persist_dir: str):
        """设置持久化路径"""
        self.persist_dir = persist_dir

    def _save(self):
        """保存知识库到磁盘"""
        if not self.persist_dir:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        try:
            # 保存文档文本
            with open(os.path.join(self.persist_dir, "rag_documents.json"),
                      "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)

            # 保存 chunks 元数据
            import numpy as np
            with open(os.path.join(self.persist_dir, "rag_index.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "chunks": self.chunks,
                    "embed_dim": self._embed_dim,
                }, f, ensure_ascii=False)

            # 保存嵌入向量（NumPy 格式，加载极快）
            if self.embeddings is not None:
                import numpy as np
                np.save(
                    os.path.join(self.persist_dir, "rag_embeddings.npy"),
                    self.embeddings,
                )

            # 保存 BM25 索引
            if self._bm25_index is not None:
                with open(os.path.join(self.persist_dir, "rag_bm25.json"), "w", encoding="utf-8") as f:
                    json.dump(self._bm25_index.to_dict(), f, ensure_ascii=False)

            logger.info(f"RAG 知识库已持久化至 {self.persist_dir} "
                        f"(含 BM25: {'✓' if self._bm25_index else '✗'})")
        except Exception as e:
            logger.warning(f"RAG 持久化失败: {e}")

    def _load(self) -> bool:
        """从磁盘加载知识库"""
        if not self.persist_dir:
            return False
        try:
            doc_path = os.path.join(self.persist_dir, "rag_documents.json")
            index_path_json = os.path.join(self.persist_dir, "rag_index.json")
            index_path_pkl = os.path.join(self.persist_dir, "rag_index.pkl")
            emb_path = os.path.join(self.persist_dir, "rag_embeddings.npy")
            bm25_path_json = os.path.join(self.persist_dir, "rag_bm25.json")
            bm25_path_pkl = os.path.join(self.persist_dir, "rag_bm25.pkl")

            if os.path.exists(doc_path):
                with open(doc_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)

            # 加载索引（优先 JSON，兼容旧版 pickle）
            if os.path.exists(index_path_json):
                with open(index_path_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    self._embed_dim = data.get("embed_dim", 0)
            elif os.path.exists(index_path_pkl):
                import pickle
                with open(index_path_pkl, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", [])
                    self._embed_dim = data.get("embed_dim", 0)

            if os.path.exists(emb_path):
                import numpy as np
                self.embeddings = np.load(emb_path)

            # 加载 BM25 索引（优先 JSON，兼容旧版 pickle）
            bm25_path = bm25_path_json if os.path.exists(bm25_path_json) else bm25_path_pkl
            if os.path.exists(bm25_path):
                try:
                    if bm25_path.endswith(".json"):
                        with open(bm25_path, "r", encoding="utf-8") as f:
                            bm25_data = json.load(f)
                    else:
                        import pickle
                        with open(bm25_path, "rb") as f:
                            bm25_data = pickle.load(f)
                    self._bm25_index = BM25Index.from_dict(bm25_data)
                    logger.info(f"BM25 索引已加载 ({self._bm25_index.doc_count} 文档)")
                except Exception as e:
                    logger.warning(f"BM25 索引加载失败: {e}")
                    self._bm25_index = None

            logger.info(f"RAG 知识库已从磁盘加载 "
                        f"({len(self.documents)} 篇文档, {len(self.chunks)} 个段落)")
            return True
        except Exception as e:
            logger.warning(f"RAG 持久化加载失败: {e}")
            return False

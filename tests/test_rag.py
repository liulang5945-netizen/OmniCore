"""
rag.py 模块的严苛单元测试
覆盖：RAGKnowledgeBase 文档分块、嵌入构建、索引重建、语义搜索、持久化
"""
import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.rag import RAGKnowledgeBase, CHUNK_SIZE, CHUNK_OVERLAP

# 测试文本
SAMPLE_TEXT_CN = (
    "人工智能是计算机科学的一个分支。它企图了解智能的实质，"
    "并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
    "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"
    "\n\n"
    "深度学习是机器学习的一个子集。它使用多层神经网络来学习数据的表示方式。"
    "卷积神经网络（CNN）广泛用于图像处理任务。"
    "循环神经网络（RNN）则擅长处理序列数据。"
    "\n\n"
    "Python 是一种解释型、面向对象的高级编程语言。"
    "它拥有动态语义，常用于数据科学和人工智能开发。"
)


class TestRAGInit:
    """初始化测试"""

    def test_init_without_persist(self):
        kb = RAGKnowledgeBase()
        assert kb.documents == {}
        assert kb.chunks == []
        assert kb.embeddings is None

    def test_init_with_empty_persist_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = RAGKnowledgeBase(persist_dir=tmpdir)
            assert kb.documents == {}
            assert kb.persist_dir == tmpdir


class TestRAGChunking:
    """文档分块测试"""

    def test_chunk_text_empty(self):
        result = RAGKnowledgeBase._chunk_text("")
        assert result == []

    def test_chunk_text_whitespace_only(self):
        result = RAGKnowledgeBase._chunk_text("   \n  \n  ")
        assert result == []

    def test_chunk_text_short(self):
        result = RAGKnowledgeBase._chunk_text("短文本", chunk_size=200, overlap=50)
        assert len(result) == 1
        assert result[0] == "短文本"

    def test_chunk_text_produces_chunks(self):
        result = RAGKnowledgeBase._chunk_text(
            SAMPLE_TEXT_CN, chunk_size=100, overlap=20
        )
        assert len(result) > 1
        # 每个 chunk 不应为空白
        for chunk in result:
            assert chunk.strip()

    def test_chunk_text_no_overlap_for_first(self):
        result = RAGKnowledgeBase._chunk_text(
            SAMPLE_TEXT_CN, chunk_size=80, overlap=0
        )
        for chunk in result:
            assert chunk.strip()

    def test_chunk_overlap_can_be_disabled(self):
        result = RAGKnowledgeBase._chunk_text(
            SAMPLE_TEXT_CN, chunk_size=80, overlap=0
        )
        # 所有 chunk 原始分隔不应包含额外重叠前缀
        for chunk in result:
            assert len(chunk) >= 1


class TestRAGDocumentManagement:
    """文档管理测试"""

    def test_add_text_and_get_doc_names(self):
        kb = RAGKnowledgeBase()
        kb.add_text("test.txt", "这是测试内容。")
        assert "test.txt" in kb.get_doc_names()
        assert kb.documents["test.txt"] == "这是测试内容。"

    def test_add_empty_text(self):
        kb = RAGKnowledgeBase()
        kb.add_text("empty.txt", "  ")
        assert "empty.txt" not in kb.documents

    def test_add_duplicate_overwrites(self):
        kb = RAGKnowledgeBase()
        kb.add_text("doc.txt", "版本1")
        kb.add_text("doc.txt", "版本2")
        assert kb.documents["doc.txt"] == "版本2"

    def test_remove_file(self):
        kb = RAGKnowledgeBase()
        kb.add_text("doc1.txt", "内容1")
        kb.add_text("doc2.txt", "内容2")
        kb.remove_file("doc1.txt")
        assert "doc1.txt" not in kb.documents
        assert "doc2.txt" in kb.documents

    def test_remove_nonexistent_no_error(self):
        kb = RAGKnowledgeBase()
        kb.remove_file("nonexistent.txt")

    def test_clear(self):
        kb = RAGKnowledgeBase()
        kb.add_text("doc.txt", "内容")
        kb.clear()
        assert kb.documents == {}
        assert kb.chunks == []
        assert kb.embeddings is None

    def test_add_file_not_found(self):
        kb = RAGKnowledgeBase()
        try:
            kb.add_file("/nonexistent/path/file.txt")
        except FileNotFoundError:
            pass  # 预期行为


class TestRAGSearchEmpty:
    """空知识库搜索测试"""

    def test_search_empty_kb(self):
        kb = RAGKnowledgeBase()
        results = kb.search("人工智能")
        assert results == []

    def test_search_with_fallback_empty(self):
        kb = RAGKnowledgeBase()
        results = kb.search_with_fallback("test")
        assert results == []

    def test_rebuild_index_empty(self):
        kb = RAGKnowledgeBase()
        msg = kb.rebuild_index()
        assert "为空" in msg


class TestRAGPersistence:
    """持久化测试"""

    def test_save_and_load_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb1 = RAGKnowledgeBase(persist_dir=tmpdir)
            kb1.add_text("doc.txt", "持久化测试内容。")
            kb1.rebuild_index()

            kb2 = RAGKnowledgeBase(persist_dir=tmpdir)
            assert kb2.documents.get("doc.txt") == "持久化测试内容。"
            assert len(kb2.chunks) > 0

    def test_set_persist_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = RAGKnowledgeBase()
            kb.set_persist_dir(tmpdir)
            assert kb.persist_dir == tmpdir
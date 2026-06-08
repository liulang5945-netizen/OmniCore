"""
通用知识自学习引擎 (Knowledge Learner)
========================================
平台级能力：采集 → 结构化 → 存储 → 验证 → 迭代
不绑定任何特定领域，供所有领域插件复用。

使用方式:
    from agent.knowledge_learner import KnowledgeLearner
    
    learner = KnowledgeLearner()
    learner.start_learning("量化交易基础", sources=["https://..."], depth="medium")
    result = learner.query("什么是MACD指标？")
"""
import datetime
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from core.utils import get_external_path

logger = logging.getLogger("KnowledgeLearner")


# ======================== 数据模型 ========================

@dataclass
class KnowledgeEntry:
    """单条知识条目"""
    id: str = ""
    domain: str = ""              # 所属领域
    concept: str = ""             # 核心概念/主题
    content: str = ""             # 知识内容（结构化文本）
    source: str = ""              # 来源URL/文档路径
    source_type: str = ""         # source_type: web / file / api / user
    confidence: float = 1.0       # 可信度 (0-1)
    tags: List[str] = field(default_factory=list)
    relations: List[dict] = field(default_factory=list)  # [{"subject","predicate","object"}]
    created_at: str = ""
    updated_at: str = ""
    verify_count: int = 0         # 被验证次数
    verify_correct: int = 0       # 验证正确次数
    is_outdated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeEntry":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class LearningDomain:
    """知识领域"""
    id: str = ""
    name: str = ""
    description: str = ""
    total_entries: int = 0
    verified_entries: int = 0
    coverage_score: float = 0.0    # 知识覆盖度 (0-1)
    avg_confidence: float = 0.0
    gaps: List[str] = field(default_factory=list)       # 已知知识空白
    conflicts: List[dict] = field(default_factory=list)  # 知识矛盾
    last_learned_at: str = ""
    learning_rounds: int = 0


@dataclass
class LearningSession:
    """一次学习会话"""
    session_id: str = ""
    domain: str = ""
    sources: List[str] = field(default_factory=list)
    status: str = "pending"        # pending / collecting / structuring / storing / verifying / completed / failed
    entries_collected: int = 0
    entries_new: int = 0
    entries_updated: int = 0
    entries_skipped: int = 0
    verify_score: float = 0.0      # 本轮验证得分
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    log: List[str] = field(default_factory=list)


# ======================== 知识存储 ========================

class KnowledgeStore:
    """
    知识持久化存储
    使用 JSON 文件 + 内存索引，未来可升级为向量数据库。
    """

    def __init__(self, storage_dir: str = ""):
        if not storage_dir:
            storage_dir = get_external_path("knowledge_store")
        self._dir = storage_dir
        self._entries_dir = os.path.join(self._dir, "entries")
        self._domains_dir = os.path.join(self._dir, "domains")
        self._sessions_dir = os.path.join(self._dir, "sessions")
        for d in [self._entries_dir, self._domains_dir, self._sessions_dir]:
            os.makedirs(d, exist_ok=True)

        # 内存索引
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._domains: Dict[str, LearningDomain] = {}
        self._domain_index: Dict[str, List[str]] = {}   # domain -> [entry_ids]
        self._concept_index: Dict[str, List[str]] = {}   # concept_keyword -> [entry_ids]

        self._load_all()

    def _load_all(self):
        """从磁盘加载所有数据到内存"""
        # 加载条目
        for fname in os.listdir(self._entries_dir):
            if fname.endswith(".json"):
                try:
                    path = os.path.join(self._entries_dir, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    entry = KnowledgeEntry.from_dict(data)
                    self._entries[entry.id] = entry
                    self._domain_index.setdefault(entry.domain, []).append(entry.id)
                    for word in self._tokenize(entry.concept):
                        self._concept_index.setdefault(word, []).append(entry.id)
                except Exception as e:
                    logger.warning(f"加载知识条目 {fname} 失败: {e}")

        # 加载领域
        for fname in os.listdir(self._domains_dir):
            if fname.endswith(".json"):
                try:
                    path = os.path.join(self._domains_dir, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    domain = LearningDomain(**{k: v for k, v in data.items()
                                               if k in LearningDomain.__dataclass_fields__})
                    self._domains[domain.id] = domain
                except Exception as e:
                    logger.warning(f"加载领域 {fname} 失败: {e}")

        logger.info(f"知识库加载完成: {len(self._entries)} 条知识, {len(self._domains)} 个领域")

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（中英文混合）"""
        text = text.lower().strip()
        # 英文单词
        words = re.findall(r'[a-z_]+', text)
        # 中文：按2-4字切分
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        for ch in chinese:
            for n in [2, 3, 4]:
                for i in range(len(ch) - n + 1):
                    words.append(ch[i:i+n])
        return list(set(words)) if words else [text[:10]]

    # ---- 条目 CRUD ----

    def add_entry(self, entry: KnowledgeEntry) -> str:
        """添加知识条目，返回 entry_id。若概念已存在则更新。"""
        if not entry.id:
            entry.id = str(uuid.uuid4())[:12]
        now = datetime.datetime.now().isoformat()
        if not entry.created_at:
            entry.created_at = now
        entry.updated_at = now

        # 检查是否已存在相同概念（同领域内去重）
        existing = self._find_duplicate(entry)
        if existing:
            # 合并：更新内容，提升可信度
            existing.content = entry.content if len(entry.content) > len(existing.content) else existing.content
            existing.confidence = min(1.0, (existing.confidence + entry.confidence) / 2 + 0.05)
            existing.updated_at = now
            existing.source = entry.source or existing.source
            self._save_entry(existing)
            logger.debug(f"更新已有知识: {entry.concept} (领域: {entry.domain})")
            return existing.id

        self._entries[entry.id] = entry
        self._domain_index.setdefault(entry.domain, []).append(entry.id)
        for word in self._tokenize(entry.concept):
            self._concept_index.setdefault(word, []).append(entry.id)
        self._save_entry(entry)
        logger.debug(f"新增知识: {entry.concept} (领域: {entry.domain})")
        return entry.id

    def _find_duplicate(self, entry: KnowledgeEntry) -> Optional[KnowledgeEntry]:
        """查找同领域中的重复条目"""
        domain_ids = self._domain_index.get(entry.domain, [])
        entry_words = set(self._tokenize(entry.concept))
        for eid in domain_ids:
            existing = self._entries.get(eid)
            if not existing:
                continue
            existing_words = set(self._tokenize(existing.concept))
            overlap = len(entry_words & existing_words)
            if overlap >= 2 or entry.concept.lower() == existing.concept.lower():
                return existing
        return None

    def _save_entry(self, entry: KnowledgeEntry):
        path = os.path.join(self._entries_dir, f"{entry.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        return self._entries.get(entry_id)

    def search(self, query: str, domain: str = "", top_k: int = 5) -> List[KnowledgeEntry]:
        """关键词检索知识（简单实现，后续可升级为向量检索）"""
        query_words = set(self._tokenize(query))
        scores: Dict[str, float] = {}

        for word in query_words:
            for eid in self._concept_index.get(word, []):
                entry = self._entries.get(eid)
                if not entry:
                    continue
                if domain and entry.domain != domain:
                    continue
                if entry.is_outdated:
                    continue
                scores[eid] = scores.get(eid, 0) + entry.confidence

        # 也在内容中搜索
        for eid, entry in self._entries.items():
            if domain and entry.domain != domain:
                continue
            if entry.is_outdated:
                continue
            content_lower = entry.content.lower()
            for word in query_words:
                if word in content_lower and eid not in scores:
                    scores[eid] = scores.get(eid, 0) + entry.confidence * 0.5

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self._entries[eid] for eid, _ in ranked]

    def list_domains(self) -> List[dict]:
        """列出所有领域"""
        return [asdict(d) for d in self._domains.values()]

    def get_domain(self, domain_id: str) -> Optional[LearningDomain]:
        return self._domains.get(domain_id)

    def save_domain(self, domain: LearningDomain):
        self._domains[domain.id] = domain
        path = os.path.join(self._domains_dir, f"{domain.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(domain), f, indent=2, ensure_ascii=False)

    def get_entries_by_domain(self, domain: str) -> List[KnowledgeEntry]:
        ids = self._domain_index.get(domain, [])
        return [self._entries[eid] for eid in ids if eid in self._entries]

    def get_domain_stats(self, domain: str) -> dict:
        entries = self.get_entries_by_domain(domain)
        if not entries:
            return {"domain": domain, "total": 0}
        confidences = [e.confidence for e in entries]
        verified = sum(1 for e in entries if e.verify_count > 0)
        return {
            "domain": domain,
            "total": len(entries),
            "verified": verified,
            "avg_confidence": sum(confidences) / len(confidences),
            "outdated": sum(1 for e in entries if e.is_outdated),
            "sources": list(set(e.source for e in entries if e.source)),
        }

    def save_session(self, session: LearningSession):
        path = os.path.join(self._sessions_dir, f"{session.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(session), f, indent=2, ensure_ascii=False)

    def list_sessions(self, domain: str = "") -> List[dict]:
        sessions = []
        for fname in os.listdir(self._sessions_dir):
            if fname.endswith(".json"):
                try:
                    path = os.path.join(self._sessions_dir, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if domain and data.get("domain") != domain:
                        continue
                    sessions.append(data)
                except Exception:
                    continue
        return sorted(sessions, key=lambda x: x.get("started_at", ""), reverse=True)


# ======================== 学习引擎 ========================

class KnowledgeLearner:
    """
    通用知识自学习引擎

    核心循环: 采集 → 结构化 → 存储 → 验证 → 迭代
    """

    def __init__(self, store: Optional[KnowledgeStore] = None, llm_func: Optional[Callable] = None,
                 search_func: Optional[Callable] = None, web_read_func: Optional[Callable] = None):
        """
        Args:
            store: 知识存储实例（默认自动创建）
            llm_func: LLM 调用函数，签名 func(prompt) -> str
            search_func: 搜索函数，签名 func(query) -> str
            web_read_func: 网页阅读函数，签名 func(url) -> str
        """
        self.store = store or KnowledgeStore()
        self._llm = llm_func
        self._search = search_func
        self._read_web = web_read_func
        self._fetch_fn = None  # 智能抓取函数（MCP fetch 优先）
        self._sessions: Dict[str, LearningSession] = {}
        self._lock = threading.Lock()

    def set_llm(self, llm_func: Callable):
        """设置 LLM 调用函数（延迟注入）"""
        self._llm = llm_func

    def set_search(self, search_func: Callable):
        self._search = search_func

    def set_web_reader(self, web_read_func: Callable):
        self._read_web = web_read_func

    def set_fetch_fn(self, fetch_func: Callable):
        """设置智能抓取函数（优先级高于 _read_web）"""
        self._fetch_fn = fetch_func

    # ======================== 主入口 ========================

    def start_learning(self, domain: str, sources: List[str] = None,
                       depth: str = "medium", max_items: int = 50) -> LearningSession:
        """
        启动一轮知识学习

        Args:
            domain: 知识领域名称
            sources: 来源列表（URL列表 / "auto" 表示自动搜索）
            depth: 学习深度 "shallow" / "medium" / "deep"
            max_items: 最大采集条目数

        Returns:
            LearningSession 学习会话结果
        """
        session = LearningSession(
            session_id=str(uuid.uuid4())[:10],
            domain=domain,
            sources=sources or [],
            status="collecting",
            started_at=datetime.datetime.now().isoformat(),
        )
        self._sessions[session.session_id] = session

        try:
            # 确保领域存在
            self._ensure_domain(domain)

            # Phase 1: 采集
            session.log.append(f"[{self._now()}] 开始采集领域知识: {domain}")
            raw_data = self._collect(domain, sources or [], depth, max_items, session)
            session.entries_collected = len(raw_data)

            # Phase 2: 结构化
            session.status = "structuring"
            session.log.append(f"[{self._now()}] 开始结构化 {len(raw_data)} 条原始数据")
            structured = self._structure(domain, raw_data, session)

            # Phase 3: 存储
            session.status = "storing"
            session.log.append(f"[{self._now()}] 开始存储 {len(structured)} 条结构化知识")
            new_count, update_count, skip_count = self._store_entries(domain, structured, session)
            session.entries_new = new_count
            session.entries_updated = update_count
            session.entries_skipped = skip_count

            # Phase 4: 验证
            session.status = "verifying"
            session.log.append(f"[{self._now()}] 开始验证知识质量")
            verify_score = self._verify(domain, session)
            session.verify_score = verify_score

            # Phase 5: 更新领域统计
            self._update_domain_stats(domain)

            session.status = "completed"
            session.finished_at = datetime.datetime.now().isoformat()
            session.log.append(
                f"[{self._now()}] 学习完成! 新增:{new_count}, 更新:{update_count}, "
                f"跳过:{skip_count}, 验证得分:{verify_score:.1%}"
            )
            logger.info(f"学习完成 [{domain}]: {new_count} 新, {update_count} 更新")

        except Exception as e:
            session.status = "failed"
            session.error = str(e)
            session.finished_at = datetime.datetime.now().isoformat()
            session.log.append(f"[{self._now()}] ❌ 学习失败: {e}")
            logger.error(f"学习失败 [{domain}]: {e}")

        self.store.save_session(session)
        return session

    # ======================== Phase 1: 采集 ========================

    def _collect(self, domain: str, sources: List[str], depth: str,
                 max_items: int, session: LearningSession) -> List[dict]:
        """采集原始知识数据"""
        raw_data = []

        if sources and sources != ["auto"]:
            # 从指定来源采集
            for src in sources[:max_items]:
                try:
                    if src.startswith("http"):
                        content = self._fetch_web(src)
                        if content:
                            raw_data.append({"source": src, "content": content, "type": "web"})
                            session.log.append(f"  ✓ 已采集: {src[:80]}")
                    elif os.path.exists(src):
                        content = self._read_file(src)
                        if content:
                            raw_data.append({"source": src, "content": content, "type": "file"})
                            session.log.append(f"  ✓ 已采集文件: {src}")
                except Exception as e:
                    session.log.append(f"  ✗ 采集失败 {src[:60]}: {e}")
        else:
            # 自动搜索
            if self._search:
                queries = self._generate_search_queries(domain, depth)
                for query in queries:
                    try:
                        result = self._search(query)
                        if result:
                            raw_data.append({"source": f"search:{query}", "content": result, "type": "search"})
                            session.log.append(f"  ✓ 搜索: {query}")

                            # 深入读取搜索结果中的 URL
                            import re
                            urls = re.findall(r'https?://[^\s\)\"\']+', result)
                            for url in urls[:3]:  # 每个搜索结果最多深入 3 个 URL
                                try:
                                    deep_content = self._fetch_web(url)
                                    if deep_content and len(deep_content) > 100:
                                        raw_data.append({"source": url, "content": deep_content, "type": "web"})
                                        session.log.append(f"  ✓ 深入读取: {url[:60]}")
                                except Exception as e:
                                    session.log.append(f"  ✗ 读取失败 {url[:40]}: {e}")
                    except Exception as e:
                        session.log.append(f"  ✗ 搜索失败: {e}")
                    if len(raw_data) >= max_items:
                        break
            else:
                session.log.append("  ⚠ 未配置搜索函数，无法自动采集")

        return raw_data[:max_items]

    def _generate_search_queries(self, domain: str, depth: str) -> List[str]:
        """根据领域生成搜索关键词"""
        base_queries = [f"{domain} 入门", f"{domain} 核心概念", f"{domain} 实战"]
        if depth == "medium":
            base_queries += [f"{domain} 进阶", f"{domain} 最佳实践"]
        elif depth == "deep":
            base_queries += [
                f"{domain} 进阶", f"{domain} 最佳实践",
                f"{domain} 原理", f"{domain} 高级技巧",
                f"{domain} 常见问题",
            ]
        return base_queries

    def _fetch_web(self, url: str) -> str:
        """获取网页内容（优先级：_fetch_fn > _read_web > requests）"""
        # 优先使用智能抓取（MCP fetch，返回 Markdown）
        if self._fetch_fn:
            try:
                result = self._fetch_fn(url)
                if result and len(str(result).strip()) > 50:
                    return str(result)[:10000]
            except Exception:
                pass
        # 其次使用网页阅读器（read_webpage）
        if self._read_web:
            try:
                result = self._read_web(url)
                if result and len(str(result).strip()) > 50:
                    return str(result)[:10000]
            except Exception:
                pass
        # 降级：用 requests 简单获取
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)[:10000]
        except Exception:
            return ""

    def _read_file(self, path: str) -> str:
        """读取本地文件"""
        try:
            from tools.file_parser import parse_file_to_text
            return parse_file_to_text(path)
        except Exception:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()[:10000]
            except Exception:
                return ""

    # ======================== Phase 2: 结构化 ========================

    def _structure(self, domain: str, raw_data: List[dict], session: LearningSession) -> List[dict]:
        """将原始数据结构化为知识条目"""
        structured = []

        if self._llm:
            # 使用 LLM 结构化
            for item in raw_data:
                try:
                    entries = self._llm_structure(domain, item["content"], item.get("source", ""))
                    structured.extend(entries)
                    session.log.append(f"  ✓ 结构化: {item.get('source', '')[:40]} → {len(entries)} 条知识")
                except Exception as e:
                    session.log.append(f"  ✗ 结构化失败: {e}")
        else:
            # 降级：简单规则提取
            for item in raw_data:
                entries = self._rule_structure(domain, item["content"], item.get("source", ""))
                structured.extend(entries)
                session.log.append(f"  ✓ 规则提取: {item.get('source', '')[:40]} → {len(entries)} 条")

        return structured

    def _llm_structure(self, domain: str, content: str, source: str) -> List[dict]:
        """用 LLM 将文本结构化为知识条目"""
        # 截取内容避免超长
        content = content[:4000]
        prompt = f"""请将以下关于"{domain}"的文本内容提取为结构化知识条目。

每个条目包含：
1. concept: 核心概念名称
2. content: 概念的详细解释（保留原文关键信息，用自己的话概括）
3. tags: 相关标签列表
4. relations: 概念间关系列表，每项格式 {{"subject":"A","predicate":"是","object":"B"}}

请以 JSON 数组格式输出，每个条目一个对象。最多提取 5 个最重要的条目。
只输出 JSON，不要其他文字。

文本内容：
{content}
"""
        response = self._llm(prompt)
        return self._parse_llm_json(response, domain, source)

    def _parse_llm_json(self, response: str, domain: str, source: str) -> List[dict]:
        """解析 LLM 返回的 JSON"""
        # 尝试提取 JSON
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    obj = json.loads(json_match.group())
                    return [self._normalize_entry(obj, domain, source)]
                except Exception:
                    return []
            return []

        try:
            items = json.loads(json_match.group())
            if isinstance(items, dict):
                items = [items]
            return [self._normalize_entry(item, domain, source) for item in items if isinstance(item, dict)]
        except json.JSONDecodeError:
            return []

    def _normalize_entry(self, item: dict, domain: str, source: str) -> dict:
        """规范化条目格式"""
        return {
            "concept": item.get("concept", item.get("title", "未知概念")),
            "content": item.get("content", item.get("description", "")),
            "domain": domain,
            "source": source,
            "source_type": "web" if source.startswith("http") else ("search" if source.startswith("search:") else "file"),
            "tags": item.get("tags", []),
            "relations": item.get("relations", []),
            "confidence": 0.8,
        }

    def _rule_structure(self, domain: str, content: str, source: str) -> List[dict]:
        """无 LLM 时的规则提取降级方案"""
        entries = []
        # 按段落/章节切分
        paragraphs = re.split(r'\n{2,}', content)
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:
                continue
            # 尝试提取标题+内容
            lines = para.split("\n")
            title = lines[0].strip()[:100]
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else para

            # 过滤纯数字、纯标点
            if re.match(r'^[\d\s\.\-/]+$', title):
                continue

            entries.append({
                "concept": title,
                "content": body[:2000],
                "domain": domain,
                "source": source,
                "source_type": "web" if source.startswith("http") else "file",
                "tags": [],
                "relations": [],
                "confidence": 0.5,  # 规则提取可信度较低
            })

        return entries[:5]  # 限制数量

    # ======================== Phase 3: 存储 ========================

    def _store_entries(self, domain: str, structured: List[dict],
                       session: LearningSession) -> tuple:
        """存储结构化知识到知识库"""
        new_count = 0
        update_count = 0
        skip_count = 0

        for item in structured:
            if not item.get("concept") or not item.get("content"):
                skip_count += 1
                continue

            entry = KnowledgeEntry(
                domain=domain,
                concept=item["concept"],
                content=item["content"],
                source=item.get("source", ""),
                source_type=item.get("source_type", ""),
                tags=item.get("tags", []),
                relations=item.get("relations", []),
                confidence=item.get("confidence", 0.7),
            )

            existing = self.store._find_duplicate(entry)
            entry_id = self.store.add_entry(entry)

            if existing:
                update_count += 1
            else:
                new_count += 1

        return new_count, update_count, skip_count

    # ======================== Phase 4: 验证 ========================

    def _verify(self, domain: str, session: LearningSession) -> float:
        """自我验证：用已学知识回答测试题"""
        entries = self.store.get_entries_by_domain(domain)
        if not entries:
            return 0.0

        # 选择部分条目进行验证
        verify_sample = entries[:min(5, len(entries))]
        correct = 0
        total = len(verify_sample)

        for entry in verify_sample:
            if self._llm:
                # 用 LLM 生成问题并检验
                is_correct = self._llm_verify_entry(domain, entry)
            else:
                # 简单验证：内容非空且有一定长度
                is_correct = len(entry.content) > 30 and entry.confidence >= 0.5

            entry.verify_count += 1
            if is_correct:
                entry.verify_correct += 1
                correct += 1
                # 提升可信度
                entry.confidence = min(1.0, entry.confidence + 0.02)
            else:
                # 降低可信度
                entry.confidence = max(0.1, entry.confidence - 0.05)

            self.store._save_entry(entry)

        score = correct / total if total > 0 else 0.0
        session.log.append(f"  验证结果: {correct}/{total} 通过 ({score:.0%})")
        return score

    def _llm_verify_entry(self, domain: str, entry: KnowledgeEntry) -> bool:
        """用 LLM 验证单条知识"""
        try:
            prompt = f"""你是知识质量审核员。请判断以下关于"{domain}"的知识条目是否准确。

概念: {entry.concept}
内容: {entry.content[:500]}

请回答: 这条知识是否准确？只回答"正确"或"错误"，不要其他文字。"""
            response = self._llm(prompt).strip()
            return "正确" in response or "correct" in response.lower()
        except Exception:
            return True  # 验证失败默认通过

    # ======================== Phase 5: 查询 ========================

    def query(self, question: str, domain: str = "", top_k: int = 3) -> str:
        """查询已学习的知识"""
        results = self.store.search(question, domain=domain, top_k=top_k)
        if not results:
            return f"未找到与'{question}'相关的已学知识。建议先执行 learn_knowledge 学习相关领域。"

        parts = [f"📚 找到 {len(results)} 条相关知识:\n"]
        for i, entry in enumerate(results, 1):
            confidence_bar = "●" * int(entry.confidence * 5) + "○" * (5 - int(entry.confidence * 5))
            parts.append(
                f"**{i}. {entry.concept}** [{entry.domain}] (可信度: {confidence_bar} {entry.confidence:.0%})\n"
                f"   {entry.content[:300]}\n"
                f"   来源: {entry.source[:60] if entry.source else '未知'}"
            )
        return "\n\n".join(parts)

    # ======================== 辅助方法 ========================

    def _ensure_domain(self, domain_name: str):
        """确保领域存在"""
        domain_id = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '_', domain_name)[:50]
        if domain_id not in self.store._domains:
            domain = LearningDomain(
                id=domain_id,
                name=domain_name,
                description=f"知识领域: {domain_name}",
            )
            self.store.save_domain(domain)

    def _update_domain_stats(self, domain_name: str):
        """更新领域统计"""
        domain_id = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '_', domain_name)[:50]
        domain = self.store.get_domain(domain_id)
        if not domain:
            return

        entries = self.store.get_entries_by_domain(domain_name)
        domain.total_entries = len(entries)
        domain.verified_entries = sum(1 for e in entries if e.verify_count > 0)
        domain.avg_confidence = (
            sum(e.confidence for e in entries) / len(entries) if entries else 0
        )
        domain.learning_rounds += 1
        domain.last_learned_at = datetime.datetime.now().isoformat()

        # 计算覆盖度（基于验证通过率）
        if domain.verified_entries > 0:
            verified_correct = sum(
                1 for e in entries
                if e.verify_count > 0 and (e.verify_correct / e.verify_count) >= 0.6
            )
            domain.coverage_score = verified_correct / max(1, domain.total_entries)

        self.store.save_domain(domain)

    def get_learning_report(self, domain: str = "") -> str:
        """获取学习报告"""
        if domain:
            stats = self.store.get_domain_stats(domain)
            if not stats or stats["total"] == 0:
                return f"尚未学习'{domain}'相关知识。"

            sessions = self.store.list_sessions(domain)
            recent_sessions = sessions[:5]

            report = (
                f"📊 学习报告: {domain}\n"
                f"{'='*40}\n"
                f"知识条目: {stats['total']} 条\n"
                f"已验证: {stats['verified']} 条\n"
                f"平均可信度: {stats['avg_confidence']:.0%}\n"
                f"过时条目: {stats['outdated']} 条\n"
                f"数据来源: {len(stats.get('sources', []))} 个\n"
            )

            if recent_sessions:
                report += f"\n📋 最近学习记录 ({len(recent_sessions)} 次):\n"
                for s in recent_sessions:
                    status_icon = {"completed": "✅", "failed": "❌", "in_progress": "🔄"}.get(s.get("status"), "⏳")
                    report += (
                        f"  {status_icon} {s.get('started_at', '')[:16]} | "
                        f"新增:{s.get('entries_new', 0)} 更新:{s.get('entries_updated', 0)} | "
                        f"验证:{s.get('verify_score', 0):.0%}\n"
                    )
            return report

        # 所有领域汇总
        domains = self.store.list_domains()
        if not domains:
            return "尚未开始任何知识学习。"

        report = "📊 全部知识领域:\n" + "=" * 40 + "\n"
        for d in domains:
            report += (
                f"  📁 {d['name']}: {d['total_entries']} 条 | "
                f"可信度:{d['avg_confidence']:.0%} | "
                f"学习{d['learning_rounds']}轮\n"
            )
        return report

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")


# ======================== 全局单例 ========================

_knowledge_learner: Optional[KnowledgeLearner] = None


def get_knowledge_learner() -> KnowledgeLearner:
    """获取全局知识学习器单例"""
    global _knowledge_learner
    if _knowledge_learner is None:
        _knowledge_learner = KnowledgeLearner()
    return _knowledge_learner
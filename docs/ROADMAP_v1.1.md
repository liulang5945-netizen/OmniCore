# OmniCore v1.1 开发路线图

## P0 优先级
- [ ] RAG 混合检索 + 重排序
  - [ ] BM25Index 类实现
  - [ ] CrossEncoderReranker 实现
  - [ ] hybrid_search() RRF 融合
  - [ ] query_rewrite() 查询改写
  - [ ] search_with_fallback() 增强
  - [ ] 持久化 BM25 索引
  - [ ] RAG 配置 API + 前端面板
- [ ] Agent Function Calling 优化
  - [ ] ToolCallParser 多策略解析
  - [ ] FewShotGenerator
  - [ ] 增强 system_prompt
  - [ ] 自修复机制
  - [ ] 工具 Schema 完善

## P1 优先级
- [ ] LoRA/QLoRA 训练一键配置
  - [ ] TrainingRecommender
  - [ ] DatasetQualityChecker
  - [ ] GGUFExporter 完善
  - [ ] 前端训练 UI 改造
- [ ] 安全加固
  - [ ] core/security.py (JWT + 加密)
  - [ ] SecureSettingsManager
  - [ ] JWT 认证中间件
  - [ ] 沙箱资源限制增强
  - [ ] 前端安全设置面板
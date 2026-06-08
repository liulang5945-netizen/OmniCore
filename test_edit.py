import re
with open('e:/OmniCore/api/chat_strategies.py', 'r', encoding='utf-8') as f: s = f.read()
s = re.sub(r'    if hasattr\(hf_tokenizer, \x22apply_chat_template\x22\):.*?    if has_cloud_api:', '    from agent.token_optimizer import compress_history\n    compressed = compress_history(request.history, max_rounds=3, max_chars_per_round=300)\n    context_str = \x22\x22\n    if compressed:\n        context_str = (\x22【上下文】\\n\x22 + \x22\\n\x22.join(f\x22用户: {u}\\n助手: {a}\x22 for u, a in compressed) + \x22\\n\\n\x22)\n    formatted_prompt = (f\x22{request.system_prompt}\\n\\n{context_str}\x22 f\x22### Instruction:\\n{prompt}\\n### Response:\\n\x22)\n\n    if has_cloud_api:', s, flags=re.DOTALL)
with open('e:/OmniCore/api/chat_strategies.py', 'w', encoding='utf-8') as f: f.write(s)

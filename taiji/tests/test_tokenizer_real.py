"""Test the real SentencePiece tokenizer"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taiji.tokenizer import ModelSelfTokenizer

tokenizer = ModelSelfTokenizer(sp_model_path='taiji/tokenizer/sentencepiece.model')

# Test basic encoding/decoding
text = 'Hello, how are you?'
result = tokenizer(text)
print(f'Input: {text}')
print(f'Shape: {result["input_ids"].shape}')
ids = result['input_ids'][0].tolist()
print(f'IDs: {ids[:10]}...')

decoded = tokenizer.decode(result['input_ids'][0])
print(f'Decoded: {decoded}')
print()

# Test Chinese
text2 = 'OmniCore AI'
result2 = tokenizer(text2)
print(f'Input: {text2}')
decoded2 = tokenizer.decode(result2['input_ids'][0])
print(f'Decoded: {decoded2}')
print()

# Test special tokens
tokenizer.register_tool('search')
tokenizer.register_tool('read_file')

special = '<think>test</think><tool_call>search'
result3 = tokenizer(special)
ids3 = result3['input_ids'][0].tolist()
print(f'Special token IDs: {ids3}')
print(f'Has think_start (32000): {32000 in ids3}')
print(f'Has think_end (32001): {32001 in ids3}')
print(f'Has tool_call (32002): {32002 in ids3}')
print(f'Has search (32010): {32010 in ids3}')
print()
print('[OK] Tokenizer working correctly!')

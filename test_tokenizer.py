from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [{"role": "user", "content": "你好"}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("FORMATTED:")
print(repr(formatted))

# Now encode
inputs = tokenizer(formatted, return_tensors="pt")
print("INPUT IDS:")
print(inputs["input_ids"])

# Now decode
print("DECODED:")
print(tokenizer.decode(inputs["input_ids"][0]))

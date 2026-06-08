import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

def test():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    formatted = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n'
    
    inputs = tokenizer(formatted, return_tensors="pt")
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": 512,
        "streamer": streamer,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    
    outputs = []
    for text in streamer:
        outputs.append(text)
        print("YIELD:", repr(text))
        
    thread.join()
    print("TOTAL:", "".join(outputs))

test()

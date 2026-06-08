import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

def test():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    messages = [{"role": "user", "content": "你好"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": 10,
        "streamer": streamer,
    }
    
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    
    for text in streamer:
        print("STREAM YIELD:", repr(text))
        
    thread.join()

test()

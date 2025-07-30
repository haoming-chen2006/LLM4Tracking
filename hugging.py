from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# The script will automatically use the token stored by `huggingface-cli login`
# Ensure you have logged in via the terminal before running this.
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    torch_dtype=torch.float16, # Use float16 for better performance
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Example usage:
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

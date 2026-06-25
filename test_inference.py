import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model fully on GPU — no device_map splitting
base = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    trust_remote_code=True
).to("cuda")

model = PeftModel.from_pretrained(base, "./phi2-qlora-output")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

prompt = "### Instruction:\nWrite a Python function that reverses a string.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
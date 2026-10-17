import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model in 4-bit NF4...")
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print(f"Loaded base model in {time.time() - t0:.1f}s")

print("Attaching adapter phi2-qlora-output...")
model = PeftModel.from_pretrained(base_model, "phi2-qlora-output")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

prompt = "### Instruction:\nWrite a Python function that reverses a string without using slicing.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"Generating on device {model.device}...")
t1 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
gen_time = time.time() - t1
gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

print(f"Generated {len(outputs[0]) - inputs['input_ids'].shape[1]} tokens in {gen_time:.2f}s ({len(outputs[0]) - inputs['input_ids'].shape[1] / gen_time:.1f} tok/s)!")
print("Output preview:")
print(gen_text[:300])

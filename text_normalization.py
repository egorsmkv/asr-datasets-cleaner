"""
You can find other samples in the following link:

https://huggingface.co/datasets/skypro1111/ubertext-2-news-verbalized
"""

import torch

from transformers import MBartForConditionalGeneration, AutoTokenizer

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model_name = "skypro1111/mbart-large-50-verbalization"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.src_lang = "uk_XX"
tokenizer.tgt_lang = "uk_XX"

model = MBartForConditionalGeneration.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    device_map=device,
)
model.eval()

input_text = [
    "<verbalization>:Мені 23 роки і я маю $500 у своєму банку що на вул. Жмеринській!",
    "<verbalization>:Ти 5-м будеш у черзі.",
    "<verbalization>:1997 року народження, він 23 рази ходив в походи!",
]

encoded_input = tokenizer(
    input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024
).to(device)

output_ids = model.generate(
    **encoded_input, max_length=1024, num_beams=5, early_stopping=True
)
output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

for output_text in output_texts:
    print(output_text)

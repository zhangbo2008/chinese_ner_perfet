
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("ckiplab/albert-base-chinese-ner")

model = AutoModelForTokenClassification.from_pretrained("ckiplab/albert-base-chinese-ner")
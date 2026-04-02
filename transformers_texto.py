from transformers import BertTokenizer, BertModel
import torch

# Carrega modelo e tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Texto de exemplo
text = "Eu gosto de NLP"

# Tokenização
inputs = tokenizer(text, return_tensors="pt")

# Passa pelo modelo
with torch.no_grad():
    outputs = model(**inputs)

# Representações dos tokens (camada final)
last_hidden_states = outputs.last_hidden_state  # shape: [batch_size, sequence_length, hidden_size]

# Representação do [CLS] (primeiro token)
cls_embedding = last_hidden_states[:, 0, :]  # shape: [batch_size, hidden_size]

print(cls_embedding)
from torch import nn
import torch
from transformers import BertModel, BertTokenizer
from Hefesto.models.model import Model


class PreTransformersModel(Model):
    def __init__(self, input_dim, hidden_dim, n_steps):
        super().__init__(input_dim, hidden_dim, n_steps)

        # Cargar el modelo preentrenado y el tokenizador de BERT
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Capa lineal para la regresión
        self.linear = nn.Linear(
            self.bert_model.config.hidden_size, 1
        )
        # Solo una salida para la regresión

    def forward(self, input_ids, attention_mask):
        # Codificación de la entrada utilizando BERT
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        # Tomar la salida de la última capa oculta de BERT
        last_hidden_state = outputs.last_hidden_state

        # Aplicar la capa lineal para la regresión
        logits = self.linear(last_hidden_state[:, 0, :])  # Tomar solo el token [CLS]

        return logits
    
    def __str__(self):
        return "PreTransformersModel"

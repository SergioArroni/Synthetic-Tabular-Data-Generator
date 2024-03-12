from torch import nn
import torch

from Hefesto.models.model import Model


class TransformerModel(Model):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        nhead=4,
        num_encoder_layers=2,
        dropout=0.1,
    ):
        super().__init__(input_dim, hidden_dim)

        # Capa de embedding para convertir la entrada en una dimensión adecuada
        self.embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hidden_dim
        )

        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Capa de salida para clasificación
        self.output = nn.Linear(hidden_dim, input_dim)

        # Función de pérdida actualizada para clasificación (asumiendo clasificación multiclase)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src):
        # `src` es de tamaño [seq_len, batch_size], donde seq_len es la longitud de la secuencia
        embedded_src = self.embedding(src)
        transformed_output = self.transformer_encoder(embedded_src)
        output = self.output(transformed_output.mean(dim=0))
        return output

    def train_model(self, model, input, optimizer) -> torch.Tensor:
        model.train()
        optimizer.zero_grad()
        output = model(input)
        loss = self.loss_fn(
            output, input
        )  # Asumiendo que `target` es proporcionado y adecuado para CrossEntropyLoss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        return loss

    def __str__(self):
        return "TransformerModel"

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
)
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, configs):
        super(Informer, self).__init__()
        self.task_name = "classification"

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,  # 7
            configs.d_model,  # 128
            configs.embed,  # "timeF"
            configs.freq,  # "h"
            configs.dropout,  # 0.1
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            [ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)]
            if configs.distil and ("forecast" in configs.task_name)
            else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.d_model * configs.seq_len, configs.num_class
        )

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc):
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

    # def classification(self, x_enc):
    #     # enc
    #     enc_out = self.enc_embedding(x_enc, None)
    #     enc_out, attns = self.encoder(enc_out, attn_mask=None)

    #     # Output
    #     output = self.act(
    #         enc_out
    #     )  # the output transformer encoder/decoder embeddings don't include non-linearity
    #     output = self.dropout(output)
    #     output = output.reshape(
    #         output.shape[0], -1
    #     )  # (batch_size, seq_length * d_model)
    #     output = self.projection(output)  # (batch_size, num_classes)
    #     return output

    def forward(self, x_enc, x_mark_enc):
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

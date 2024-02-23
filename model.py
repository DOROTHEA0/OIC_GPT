import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")


class Attention(nn.Module):
    def __init__(self, embed_dim, n_head, bias=True, use_head_alloc=True):
        super(Attention, self).__init__()
        self.q_linear = MultiLinear(n_head, embed_dim, bias, use_head_alloc)
        self.k_linear = MultiLinear(n_head, embed_dim, bias, use_head_alloc)
        self.v_linear = MultiLinear(n_head, embed_dim, bias, use_head_alloc)
        out_dim = embed_dim if use_head_alloc else embed_dim * n_head
        self.out_linear = nn.Linear(out_dim, embed_dim)

    def forward(self, x, mask=False):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        bsz, n_head, seq_len, hidden_dim = k.shape
        qk = q @ torch.transpose(k, 2, 3)

        if mask:
            mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1)
            mask = mask.bool()
            mask = mask[None, None, :, :]
            qk = qk.masked_fill(mask, float('-inf'))

        dk = torch.tensor(hidden_dim, dtype=qk.dtype, device=qk.device)
        scores = F.softmax(qk / torch.sqrt(dk), dim=-1)
        att_out = scores @ v
        att_out = torch.permute(att_out, (0, 2, 1, 3))
        att_out = att_out.reshape(bsz, seq_len, n_head * hidden_dim)
        return self.out_linear(att_out)


class MultiLinear(nn.Module):
    def __init__(self, n_head, hidden_dim, bias=True, use_head_alloc=True):
        super(MultiLinear, self).__init__()
        head_dim = hidden_dim // n_head if use_head_alloc else hidden_dim
        self.weight = nn.Parameter(torch.empty(n_head, hidden_dim, head_dim))
        self.bias = nn.Parameter(torch.empty(n_head, 1, head_dim)) if bias else None
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = x[:, None, :, :]
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        return x


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_head, feed_hid_dim, att_bias=True, use_head_alloc=True):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(embed_dim, n_head, bias=att_bias, use_head_alloc=use_head_alloc)
        self.feedforward = FeedForward(embed_dim, feed_hid_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=True):
        h = self.layer_norm1(self.attention(x, mask) + x)
        out = self.layer_norm2(self.feedforward(h) + h)
        return out


class GenerativeTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_head, feed_hid_dim, n_layers, att_bias=True, use_head_alloc=True):
        super(GenerativeTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.feed_hid_dim = feed_hid_dim
        self.n_layers = n_layers

        self.word_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerBlock(embed_dim=self.embed_dim, n_head=self.n_head,
                                 feed_hid_dim=self.feed_hid_dim, att_bias=att_bias, use_head_alloc=use_head_alloc))
        self.out_layer = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, word_indices, mask=True):
        word_embed = self.word_embed(word_indices)
        for layer in self.layers:
            word_embed = layer(word_embed, mask)
        return self.out_layer(word_embed)


if __name__ == '__main__':
    model = GenerativeTransformer(10000, 2048, 8, feed_hid_dim=2048, n_layers=16)
    x = torch.tensor([[2, 5, 2, 7, 4, 6, 12, 65, 23, 75], [2, 5, 2, 7, 4, 6, 12, 65, 23, 75]], dtype=torch.int)
    print(x.shape)
    print(model(x).shape)
    count_parameters(model)

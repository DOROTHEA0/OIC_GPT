import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import tokenizer
import utils


class ByteEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(ByteEmbedding, self).__init__()
        embed_size = 256 * 4
        self.embed = nn.Embedding(num_embeddings=embed_size, embedding_dim=embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        x = self.embed(x)
        bsz, seq_len, c_len, embed_dim = x.shape
        x = x.reshape(bsz, seq_len, c_len * embed_dim)
        return self.proj(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, embed_dim, n_head, bias=True, use_head_alloc=True):
        super(Attention, self).__init__()
        self.q_linear = MultiLinear(n_head, embed_dim, bias=bias, use_head_alloc=use_head_alloc)
        self.k_linear = MultiLinear(n_head, embed_dim, bias=bias, use_head_alloc=use_head_alloc)
        self.v_linear = MultiLinear(n_head, embed_dim, bias=bias, use_head_alloc=use_head_alloc)
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
    def __init__(self, n_head, hidden_dim, out_dim=None, bias=True, use_head_alloc=True):
        super(MultiLinear, self).__init__()
        head_dim = hidden_dim // n_head if use_head_alloc else hidden_dim
        if out_dim is not None:
            head_dim = out_dim

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
        self.layer_norm1 = RMSNorm(embed_dim)
        self.layer_norm2 = RMSNorm(embed_dim)

    def forward(self, x, mask=True):
        h = self.attention(self.layer_norm1(x), mask) + x
        out = self.layer_norm2(self.feedforward(h) + h)
        return out


class GenerativeTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_head, feed_hid_dim, n_layers, att_bias=False, use_head_alloc=True, use_byte_embed=False):
        super(GenerativeTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.feed_hid_dim = feed_hid_dim
        self.n_layers = n_layers
        self.use_byte_embed = use_byte_embed

        self.word_embed = ByteEmbedding(embed_dim=self.embed_dim) if use_byte_embed else nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerBlock(embed_dim=self.embed_dim, n_head=self.n_head,
                                 feed_hid_dim=self.feed_hid_dim, att_bias=att_bias, use_head_alloc=use_head_alloc))
        self.norm = RMSNorm(self.embed_dim)
        self.out_layer = MultiLinear(n_head=4, hidden_dim=self.embed_dim, out_dim=256) if use_byte_embed else nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, word_indices, mask=True):
        word_embed = self.word_embed(word_indices)
        for layer in self.layers:
            word_embed = layer(word_embed, mask)
        out_logits = self.out_layer(self.norm(word_embed))
        if self.use_byte_embed:
            out_logits = torch.transpose(out_logits, 1, 2)
        return out_logits

class SSM(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(SSM, self).__init__()
        self.A = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.B = nn.Parameter(torch.empty(embed_dim, hidden_dim))
        self.C = nn.Parameter(torch.empty(hidden_dim, embed_dim))
        self.D = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.hidden_state = None

    def forward(self, x):
        bsz, seq_len, embed_dim = x.shape
        if self.hidden_state is None:
            self.hidden_state = nn.Parameter(torch.empty(bsz, seq_len, embed_dim, embed_dim))
        x_ = x @ self.B
        # 没写完
        pass



if __name__ == '__main__':
    model = GenerativeTransformer(1, 512, 8, 512, 6, att_bias=False, use_head_alloc=True, use_byte_embed=True)
    utils.test_model(model)

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.q_linear = ParallelLinear(n_head, embed_dim, bias, use_head_alloc)
        self.k_linear = ParallelLinear(n_head, embed_dim, bias, use_head_alloc)
        self.v_linear = ParallelLinear(n_head, embed_dim, bias, use_head_alloc)
        out_dim = embed_dim if use_head_alloc else embed_dim * n_head
        self.out_linear = nn.Linear(out_dim, embed_dim)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        bsz, n_head, seq_len, hidden_dim = k.shape
        qk = q @ torch.transpose(k, 2, 3)
        dk = torch.tensor(hidden_dim, dtype=qk.dtype, device=qk.device)
        scores = F.softmax(qk / torch.sqrt(dk), dim=-1)
        att_out = scores @ v
        print(att_out.shape)
        att_out = torch.permute(att_out, (0, 2, 1, 3))
        print(att_out.shape)
        att_out = att_out.reshape(bsz, seq_len, n_head * hidden_dim)
        return self.out_linear(att_out)


class ParallelLinear(nn.Module):
    def __init__(self, n_head, hidden_dim, bias=True, use_head_alloc=True):
        super(ParallelLinear, self).__init__()
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

if __name__ == '__main__':
    m = Attention(128, 8)
    x = torch.rand((2, 7, 128))
    print(m(x).shape)
    count_parameters(m)
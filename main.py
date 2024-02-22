import torch
import torch.nn as nn
import torch.onnx

model = nn.MultiheadAttention(embed_dim=16, num_heads=8, dropout=0.)
nn.LayerNorm
x = torch.randn((1, 10, 4))

onnx_filename = "model_with_attention.onnx"
torch.onnx.export(model, (x, x, x), onnx_filename, verbose=True)


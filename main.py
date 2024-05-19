from tokenizer import OICTokenizer
import torch.onnx
from model import GenerativeTransformer

if __name__ == '__main__':
    m = GenerativeTransformer(vocab_size=256, embed_dim=512, n_head=8, feed_hid_dim=512, n_layers=6, use_byte_embed=True)
    m.load_state_dict(torch.load("model.pth"))
    t = OICTokenizer()
    text = "一般由："
    x, _ = t.encode(text)
    x = torch.tensor(x, dtype=torch.int)[None, :, :]
    for _ in range(100):
        with torch.no_grad():
            out_logits = m(x)
            _, indices = torch.max(out_logits, dim=-1)
            indices = indices[:, -1, :]
            next = indices + torch.tensor([[0, 256, 256 * 2, 256 * 3]], dtype=torch.int)
            x = x.tolist()
            next = next.tolist()
            x[0].append(next[0])
            x = torch.tensor(x, dtype=int)
            print(t.decode(indices.tolist()), end="")




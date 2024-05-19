from typing import Tuple

import torch
from torch import optim, nn
from tokenizer import OICTokenizer

def test_model(model):
    model = model.cuda()
    text = '''三方协议是什么？请回答：
    三方协议是指一份协议需要三个参与主体。通俗的讲就是三个方面。这三方协议通常是指毕业生在找工作时需要签署的一份协议。一般由：校方，用人单位，还有自己（找工作的人）构成。三方协议是国家为了保护毕业生就业，统计就业人数等要求的一个东西。此类协议如果是依法签订的，并且不存在法定合同无效的情形，就是合法有效的。'''
    t = OICTokenizer()
    x, x_ = t.encode(text)
    x, x_ = torch.tensor(x, dtype=torch.int)[None, :, :], torch.tensor(x_, dtype=torch.int)
    y = x_[1:, :].clone().view(-1).cuda()
    x_train = x[:, :len(text) - 1, :].cuda()
    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=10e-5, lr=3e-4)
    criterion = nn.CrossEntropyLoss()



    epochs = 1000

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        bsz, seq_l, c_l, l_l = output.shape
        output = output.reshape(bsz, seq_l * c_l, l_l)
        loss = criterion(output.view(-1, output.shape[-1]), y.long())
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    torch.save(model.state_dict(), 'model.pth')


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

import torch
from torch import optim, nn
from tokenizer import OICTokenizer

def test_model(model):
    model = model.cuda()
    text = "茄子茄子茄子茄子茄子茄子茄子茄子"
    t = OICTokenizer()
    x, x_ = t.encode(text)
    x, x_ = torch.tensor(x, dtype=torch.int)[None, :, :], torch.tensor(x_, dtype=torch.int)
    y = x_[1:, :].clone().view(-1).cuda()
    x_train = x[:, :len(text) - 1, :].cuda()
    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=10e-5, lr=3e-4)
    criterion = nn.CrossEntropyLoss()



    epochs = 200

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
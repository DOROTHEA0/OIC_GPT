import torch
from torch import optim, nn


def test_model(model):
    batch_size = 64
    seq_len = 10
    embed_dim = 128
    x_train = torch.rand((batch_size, seq_len, embed_dim), dtype=torch.float)
    y_train = x_train.clone() * 2
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 5000

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    with torch.no_grad():
        test_output = model(x_train)
        test_loss = criterion(test_output, y_train)
        print(f'Test Loss: {test_loss.item()}')
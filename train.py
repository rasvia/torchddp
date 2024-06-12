import torch


def training_loop(model, train_loader, lossFn, optimizer, epoch):
    print('Start training.....')
    for i in range(epoch):
        running_loss = 0.0
        for image, label in train_loader:
            output = model(image)
            loss = lossFn(output, label)
            loss.mean().backward()
            optimizer.step()

            running_loss += loss.mean().item()

        running_loss /= len(train_loader)
        print(f'[{i + 1}/{epoch}] loss: {running_loss:.4f}')
    print('Finished Training, and saving model...', end= ' ')
    torch.save(model.state_dict(), './model.pth')
    print('Done.')

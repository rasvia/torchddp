import torch
import torch.nn.functional as F
import torch.distributed as dist


def training_loop(model, train_loader, lossFn, optimizer, epoch, rank, world_size):
    if rank == 0:
        print('Start training.....')
    model.train()
    for i in range(epoch):
        running_loss = 0.0
        train_loader.sampler.set_epoch(i)

        for idx, (image, label) in enumerate(train_loader):
            image = image.type(torch.FloatTensor)
            output = model(image)
            label = F.one_hot(label, num_classes=10).to(output.device)
            loss = lossFn(output, label.float())
            loss.mean().backward()
            optimizer.step()

            running_loss += loss.mean().item()

        running_loss /= len(train_loader)

        train_loss = torch.Tensor([running_loss]).cuda()
        # Then, you perform the reduction (SUM in this case) across all devices
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss /= world_size

        train_loss = train_loss.item()
        if rank == 0:
            print(f'[{i + 1}/{epoch}] loss: {train_loss:.4f}')
    if rank == 0:
        print('Finished Training, and saving model...', end=' ')
        torch.save(model.state_dict(), './model.pth')
        print('Done.')

import torch

def cuda_available(nn):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            nn = torch.nn.DataParallel(nn)
        nn.cuda()
    else:
        nn.cpu()

    return nn

import torch
import torch.utils.data
import torchvision


def get_dataloader(bsz: int):
    train_data = torchvision.datasets.CIFAR10(root='./data', 
                                          train=True, 
                                          transform=torchvision.transforms.ToTensor(), # avoid getting PIL images
                                          download=True)
    test_data = torchvision.datasets.CIFAR10(root='./data', 
                                         train=False, 
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
    
    
    
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=bsz, 
                                               shuffle=False,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=bsz,
                                              shuffle=False,
                                              num_workers=2)
    return train_loader, test_loader





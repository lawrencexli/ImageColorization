import torchvision
from copy import deepcopy
from tqdm import tqdm
from dataset import *
from model import * 
import pickle

size = 256
mirflickr = Mirflickr()
train_dataloader, eval_dataloader = mirflickr.build_dataset(size=size, batch_size=16)

model = MainModel(size=size, pretrained=False)

def Unet_train(net_G, train_dl, opt, criterion, epochs, device):
    performance = []
    for e in range(epochs):
        epoch_loss = 0.0
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss) * 100.
            
        performance.append(epoch_loss / len(train_dl.dataset))
        print("=> Epoch %d/%d: L1 Loss = %.4f" % (e+1, epochs, epoch_loss / len(train_dl.dataset)))
    return net_G, performance
        
net_G = model.build_ResUNet(1, 2, size)
device = model.device
opt = torch.optim.Adam(net_G.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()        
net_G, performance = Unet_train(net_G, train_dataloader, opt, criterion, 20, device)
torch.save(net_G.state_dict(), "saved_weights/res18-unet.pt")

with open('saved_weights/performance-unet.pkl', 'wb') as f:
    pickle.dump(performance, f)
"""
Training function.
Code structure and idea from
https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
"""

import torchvision
from torchmetrics.functional import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from dataset import *
from model import *
import pickle
from skimage.metrics import structural_similarity as ssim

size = 256
mirflickr = Mirflickr()
train_dataloader, eval_dataloader = mirflickr.build_dataset(size=size, batch_size=16)
model = MainModel(size=size, pretrained=True)

def average(lst):
    return sum(lst) / len(lst)

def l1_penalty(epoch):
    denominator = 1/99 + 1.05**(epoch - 200)
    return 1/denominator + 1

"""
Compute SSIM similarity score between generated ab channel and true ab channel
"""
def compute_ssim(true, result, is_train=False):
    if is_train:
        true = true.detach().cpu().numpy()
        result = result.detach().cpu().numpy()
        
        a = ssim(true[:, 0, :, :], result[:, 0, :, :], data_range=true.max() - true.min())
        b = ssim(true[:, 1, :, :], result[:, 1, :, :], data_range=true.max() - true.min())
    else:
        true = true.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        result = result.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
        a = ssim(true[:, :, 0], result[:, :, 0], data_range=true.max() - true.min())
        b = ssim(true[:, :, 1], result[:, :, 1], data_range=true.max() - true.min())
        
    return (a + b) / 2

"""
Main GAN train function
"""
def train(model, train_dataloader, num_epoch):
    performance = {'d_fake_loss': [], # The loss between fake image and 0 (False)
                   'd_real_loss': [], # The loss between real image and 1 (True)
                   'd_loss': [],      # d_fake_loss + d_real_loss
                   'g_gan_loss': [],  # The loss between fake image and 1 (True) !!So the generator can generate realistic fake images!!
                   'g_l1_loss': [],   # The loss between the generated fake color and actual color 
                   'ssim': [],        # SSIM evaluation metric
                   'psnr': []}        # PSNR evaluation metric
     
    #l1_loss_penalty = 100.
    for epoch in range(num_epoch):
        total_d_fake_loss = 0.0
        total_d_real_loss = 0.0
        total_d_loss = 0.0
        total_g_gan_loss = 0.0
        total_g_l1_loss = 0.0
        total_g_loss = 0.0
        ssim = []
        psnr = []
        #GAN_loss_count = 0
        l1_loss_penalty = l1_penalty(epoch)
        
        for data in tqdm(train_dataloader):
            L = data['L'].to(model.device)
            ab = data['ab'].to(model.device)
            fake_outputs = model.netG(L)
            
            #####
            # Train discriminator
            # Get a fake image from generator, feed it to discriminator for loss
            #####
            model.netD.train()
            
            for p in model.netD.parameters():
                p.requires_grad = True
                
            model.optimD.zero_grad()
            
            fake_image = torch.cat([L, fake_outputs], dim=1)
            fake_preds = model.netD(fake_image.detach()) # Detach the gradients from generator
            false_label = torch.tensor(0).expand_as(fake_preds).float().to(model.device)
            fake_loss = model.DLoss(fake_preds, false_label) # Label: False
            total_d_fake_loss += float(fake_loss) * 100
            
            real_image = torch.cat([L, ab], dim=1)
            real_preds = model.netD(real_image)
            true_label = torch.tensor(1).expand_as(real_preds).float().to(model.device)
            real_loss = model.DLoss(real_preds, true_label) # Label: True
            total_d_real_loss += float(real_loss) * 100
            
            d_loss = (fake_loss + real_loss) * 0.5
            d_loss.backward()
            model.optimD.step() # Update weights
            total_d_loss += float(d_loss) * 100
            
            #####
            # Train generator
            #####
            model.netG.train()
            
            for p in model.netD.parameters():
                p.requires_grad = False
                
            model.optimG.zero_grad()
            
            fake_image = torch.cat([L, fake_outputs], dim=1)
            fake_preds = model.netD(fake_image)
            true_label = torch.tensor(1).expand_as(fake_preds).float().to(model.device)
            g_gan_loss = model.DLoss(fake_preds, true_label) # Minimize this loss for better creation of realistic fake images
            total_g_gan_loss += float(g_gan_loss) * 100
            
            g_l1_loss = model.GLoss(fake_outputs, ab) * l1_loss_penalty # Multiplied by a L1 regularization term to balance the L1 loss and GAN loss. Prioritize the L1 loss over GAN loss.
            total_g_l1_loss += float(g_l1_loss) 
            
            # Loss for fooling discriminator and loss for the differences between generated color and true color
            g_loss = g_gan_loss + g_l1_loss
            g_loss.backward()
            model.optimG.step()
            
            psnr.append(float(peak_signal_noise_ratio(ab, fake_outputs)))
            ssim.append(float(compute_ssim(ab, fake_outputs, is_train=True)))
            
        # Record all loss and accuracy metrics
        performance['d_fake_loss'].append(total_d_fake_loss / len(train_dataloader.dataset))
        performance['d_real_loss'].append(total_d_real_loss / len(train_dataloader.dataset))
        performance['d_loss'].append(total_d_loss / len(train_dataloader.dataset))
        performance['g_gan_loss'].append(total_g_gan_loss / len(train_dataloader.dataset))
        performance['g_l1_loss'].append(total_g_l1_loss / len(train_dataloader.dataset))
        performance['ssim'].append(average(ssim))
        performance['psnr'].append(average(psnr))
        
        # Perform L1 regularization decrease for G L1 loss
        """
        if epoch >= 25 and performance['g_gan_loss'][-1] > performance['g_gan_loss'][-2]:
            GAN_loss_count += 1
            
            if GAN_loss_count == 5:
                l1_loss_penalty -= 5.
                if l1_loss_penalty < 1.:
                    l1_loss_penalty = 1.
            
        else:
            GAN_loss_count = 0
        """
        
        print('=> Current generator L1 loss penalty factor: %.1f' % (l1_loss_penalty))
        print('=> Epoch %d/%d: L1 loss = %.4f, GAN loss = %.4f, SSIM = %.4f, PSNR = %.4f ' % (epoch+1, 
                                                                                              num_epoch, 
                                                                                              total_g_l1_loss / len(train_dataloader.dataset),
                                                                                              total_g_gan_loss / len(train_dataloader.dataset),
                                                                                              average(ssim),
                                                                                              average(psnr)))
    
    #####
    # Always return the generator model for final evaluation and inference
    #####
    return model.netG, performance

model_result, performance = train(model, train_dataloader, num_epoch=120)
torch.save(model_result, "saved_weights/unet_GAN_25k")

with open('saved_weights/performance_25k.pkl', 'wb') as f:
    pickle.dump(performance, f)

"""
Model function.
Code structure and idea from
https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
"""

import torchvision
import torch
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet


class MainModel:
    
    def __init__(self, size=256, pretrained=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if pretrained:
            # Build and load pretrained UNet weights
            self.netG = self.build_ResUNet(1, 2, size)
            self.netG.load_state_dict(torch.load("saved_weights/res18-unet.pt", map_location=self.device))
        else:
            # Build a new UNet and randomly initialize weights for it
            self.netG = self.init_weights(self.build_ResUNet(1, 2, size))
            
        # Build a new discriminator and randomly initialize weights for it
        self.netD = self.init_weights(self.build_batch_discriminator())
        
        # Define loss
        self.GLoss = torch.nn.L1Loss()
        self.DLoss = torch.nn.BCELoss()
        
        # Define optimizer
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
        
    """
    Build a UNet with Resnet18 as backbone
    Directly from https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
    """
    def build_ResUNet(self, num_input, num_output, size):
        resnet18 = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        body = create_body(resnet18, pretrained=True, n_in=num_input, cut=-2)
        net = DynamicUnet(body, num_output, (size, size))
        return net.to(self.device)
    
    """
    Build a batch discriminator. Directly implemented the DCGAN paper from google cloud tutorial
    """
    def build_batch_discriminator(self):
        d_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )
        return d_net.to(self.device)
    
    """
    Initialize net weights using normal distribution
    Implemented the method mentioned in UNet and DCGAN paper
    Directly from https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
    """
    def init_weights(self, net, gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and 'Conv' in classname:
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=gain)
                
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
                    
            elif 'BatchNorm2d' in classname:
                torch.nn.init.normal_(m.weight.data, 1., gain)
                torch.nn.init.constant_(m.bias.data, 0.)

        net.apply(init_func)
        return net
    


        
    
    
        
    
    
        
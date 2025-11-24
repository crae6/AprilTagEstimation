import os,argparse
import numpy as np
from PIL import Image
from FFA import *
import torch
import cv2
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dcp import *
from retinex import *

abs=os.getcwd()+'/'
def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

def init_model():
    model_dir=abs+f'trained_models/ots_train_ffa_3_19.pk'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ckp=torch.load(model_dir,map_location=device,weights_only=False)
    net=FFA(gps=3,blocks=19)
    net=nn.DataParallel(net)
    net.load_state_dict(ckp['model'])
    net.eval()
    return net

def ffa_dehaze(img, net):
    # haze = Image.open(img)
    haze = img
    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    haze_no=tfs.ToTensor()(haze)[None,::]
    with torch.no_grad():
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    # tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
    # vutils.save_image(ts,output_dir+im.split('.')[0]+'_FFA.png')
    img = ts.detach().cpu().numpy()                 # (3, H, W)
    img = np.transpose(img, (1, 2, 0))                 # (H, W, 3)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return ts, img

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--img',type=str,default='canon.jpg',help='image path')
    opt=parser.parse_args()
    net = init_model()
    img_path = 'canon.jpg'
    img = cv2.imread('canon.jpg')

    dehaze, dehaze_cv = ffa_dehaze(img, net)
    # uint8 HWC
    img_dcp = dcp(img)
    img_ssr = SSR(img)
    img_MSR = MSR(img)
    img = cv2.imread('canon.jpg')

    imgs = [img, img_dcp, img_ssr, img_MSR, dehaze_cv]   # each should be H×W×3 in RGB
    titles = ['Original', 'DCP', 'Retinex SSR', 'Reinex MSR', 'FFA']

    plt.figure(figsize=(16, 4))
    for i, (im, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 5, i+1)
        plt.imshow(im)
        plt.title(title)
        plt.axis('off')

    plt.show()
    # cv2.imshow('dehazed', dehaze_cv)
    # cv2.waitKey()
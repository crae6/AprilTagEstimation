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
from pupil_apriltags import Detector

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

def draw_apriltag_detections(img, detections, color=(0, 255, 0)):
    """
    Draw AprilTag bounding boxes and centers on an image.
    img: uint8 BGR image
    detections: output of detector.detect()
    """
    out = img.copy()

    for det in detections:
        corners = det.corners.astype(int)

        # Draw box
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(out, p1, p2, color, 10)

        # Draw center
        cx, cy = map(int, det.center)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        # Draw tag ID
        cv2.putText(
            out,
            f"ID {det.tag_id}",
            (corners[0][0], corners[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    return out


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

def processed2gray(img):
    gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    gray = gray.astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return gray 

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

    print('init model')
    net = init_model()
    img_path = '../dataset/pi_cam/pi_cam_18_haze.jpg'
    # img_path = 'canon.jpg'

    img = cv2.imread(img_path)
    print('Processing...')
    scale = 0.25
    ffa_img = cv2.resize(img, None, fx=scale, fy=scale)
    print()
    dehaze, dehaze_cv = ffa_dehaze(ffa_img, net)
    print('model done')
    # uint8 HWC
    img_dcp = dcp(img)
    
    print('dcp done')
    img_ssr = SSR(img, variance=30)
    print('ssr done')
    img_MSR = MSR(img, variance_list=[15, 80])
    print('msr done')
    img = cv2.imread(img_path)

    imgs = [img, img_dcp, img_ssr, img_MSR, dehaze_cv]   # each should be H×W×3 in RGB
    # imgs = [img, img_dcp, img_ssr, img_MSR]   # each should be H×W×3 in RGB

    titles = ['Original', 'DCP', 'Retinex SSR', 'Reinex MSR', 'FFA']

    base_detection = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    dcp_detection = detector.detect(processed2gray(img_dcp))
    ssr_detection = detector.detect(processed2gray(img_ssr))
    msr_detection = detector.detect(processed2gray(img_MSR))
    ffa_detection = detector.detect(processed2gray(dehaze_cv))

    detections_list = [base_detection, dcp_detection, ssr_detection, msr_detection, ffa_detection]

    plt.figure(figsize=(16, 4))
    for i, (im, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 5, i+1)
        # plt.imshow(im)
        plt.imshow(draw_apriltag_detections(im, detections_list[i]))
        plt.title(title)
        plt.axis('off')

    plt.show()
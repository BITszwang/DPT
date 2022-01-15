import time
import argparse
from scipy.io import savemat
import torch.backends.cudnn as cudnn
from utils import *

from model_DPT import Net



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument('--testset_dir', type=str, default='/media/wsz/FAB9B702EAEE0235/szwang/test/DPTTestingData/4xSR_5x5/')

    parser.add_argument("--patchsize", type=int, default=128, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=64, help="The stride between two test patches is set to patchsize/2")

    parser.add_argument('--model_path', type=str, default='/media/wsz/FAB9B702EAEE0235/szwang/test/DPT_private/log/DPT_X4.pth.tar')
    parser.add_argument('--save_path', type=str, default='/media/wsz/FAB9B702EAEE0235/szwang/test/DPT_private/Results/DPT-X4/')

    parser.add_argument('--psnr_ssim_save_path', type=str, default='/media/wsz/FAB9B702EAEE0235/szwang/test/DPT_private/PSNRSSIM/DPT-X4/')

    return parser.parse_args()


def test(cfg, test_Names, test_loaders):

    net = Net(cfg.angRes, cfg.upscale_factor)
    net.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):

        if cfg.upscale_factor == 2:
        ## if you perform x2 SR, just use
            model = torch.load(cfg.model_path, map_location={'cuda:1': cfg.device})
        else:
        ## if you perform x4 SR, just use
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})

        net.load_state_dict(model['state_dict'])
    else:
        print("=> no model found at '{}'".format(cfg.load_model))

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test = inference(test_loader, test_name, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass
        pass


def inference(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angRes, vw // cfg.angRes
        subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.stride)  # numU, numV, h*angRes, w*angRes
        numU, numV, H, W = subLFin.shape
        subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize * cfg.upscale_factor, cfg.angRes * cfg.patchsize * cfg.upscale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(cfg.device))
                    subLFout[u, v, :, :] = out.squeeze()

        outLF = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.upscale_factor, cfg.stride * cfg.upscale_factor, h0 * cfg.upscale_factor, w0 * cfg.upscale_factor)

        file_name= cfg.psnr_ssim_save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '-PNSR-SSIM.mat' # new added
        psnr, ssim = test_cal_metrics(label, outLF, cfg.angRes,file_name)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        isExists = os.path.exists(cfg.save_path + test_name)
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name)

        savemat(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                        {'LF': outLF.numpy()})
        pass


    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test


def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    test(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

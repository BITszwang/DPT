import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)


        x_mean = (x0+x1+x2)/3
        x = torch.cat([x_mean,x_mean,x_mean],dim=1)

        return x



class getgradient_function(nn.Module):
    def __init__(self):
        super(getgradient_function, self).__init__()
        self.gradient = Get_gradient()

    def forward(self,x):
        x = x.unsqueeze(0)
        out = self.gradient(x)
        return out





def main():

    getgradient = getgradient_function()


    method = 'DPT-X2'  # DPT-X4 HR
    input_root_path = '/media/wsz/FAB9B702EAEE0235/szwang/test/DPT_private/SRimages'
    save_root_path = '/media/wsz/FAB9B702EAEE0235/szwang/test/DPT_private/GRAimages'


    allresults_path = os.path.join(input_root_path, method)
    gradresults_path = os.path.join(save_root_path, method)

    if os.path.exists(gradresults_path):
        print('next')
    else:
        os.mkdir(gradresults_path)


    datset_names = sorted(os.listdir(allresults_path))
    dataset_num = len(datset_names)

    for i in range(dataset_num):
        dataset_result_path = os.path.join(allresults_path,datset_names[i])
        save_result_path = os.path.join(gradresults_path,datset_names[i])

        if os.path.exists(save_result_path):
            print('next')
        else:
            os.mkdir(save_result_path)

        scene_names = sorted(os.listdir(dataset_result_path))
        scene_num = len(scene_names)

        for j in range(scene_num):
            img_path = os.path.join(dataset_result_path,scene_names[j])
            grad_path = os.path.join(save_result_path,scene_names[j])

            if os.path.exists(grad_path):
                print('next')
            else:
                os.mkdir(grad_path)

            img_names = sorted(os.listdir(img_path))
            img_num = len(img_names)

            for k in range(img_num):
                image_path = os.path.join(img_path, img_names[k])
                gradient_path = os.path.join(grad_path, img_names[k])
                img = cv2.imread(image_path)
                img = img.astype(np.float32) / 255.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor_img = torch.from_numpy(np.transpose(img, (2, 0, 1))).cuda()
                tensor_img = tensor_img.float()
                gradient = getgradient(tensor_img)
                gradient = gradient.squeeze(0).cpu()
                gradient = np.transpose(gradient.numpy(), (1, 2, 0))
                gradient = gradient * 255.
                cv2.imwrite(gradient_path,gradient)


if __name__ == "__main__":
    main()
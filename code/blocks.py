import torch
import torch.nn as nn
import torch.nn.functional as F




def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings



class CrossAttentionSALSA(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=4, softmax_scale=10, shape=64, p_len=64, in_channels=64
                 , inter_channels=16, use_multiple_size=False, use_topk=False, add_SE=False):
        super(CrossAttentionSALSA, self).__init__()
        self.ksize = ksize
        self.shape = shape
        self.p_len = p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size = use_multiple_size
        self.use_topk = use_topk
        self.add_SE = add_SE
        # self.SE=SE_net(in_channels=in_channels)
        self.conv33 = nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                padding=0)
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, s,g):
        # s content
        # g gradient

        output = []

        for B in range(s.shape[0]):  # for each batch
            b_one = s[B]
            d_one = g[B]

            kernel = self.ksize

            # spatial-angular convolutional tokenization
            b1 = self.g(b_one)
            b2 = self.theta(d_one)
            b3 = self.phi(d_one)

            raw_int_bs = list(b1.size())


            patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                          strides=[self.stride_1, self.stride_1],
                                                          rates=[1, 1],
                                                          padding='same')
            patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel,
                                     -1)
            patch_28 = patch_28.permute(0, 4, 1, 2, 3)

            patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                            strides=[self.stride_2, self.stride_2],
                                                            rates=[1, 1],
                                                            padding='same')
            patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel,
                                       -1)
            patch_112 = patch_112.permute(0, 4, 1, 2, 3)

            patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                                strides=[self.stride_2, self.stride_2],
                                                                rates=[1, 1],
                                                                padding='same')
            patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel,
                                           -1)
            patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)

            _, paddings = same_padding(torch.split(b3, 1, dim=0)[0], [self.ksize, self.ksize], [1, 1],
                                       [1, 1])


            # spatial-angular self-attention
            q = patch_28.contiguous().view(patch_28.shape[0] * patch_28.shape[1], -1)
            k = patch_112_2.permute(2, 3, 4, 0, 1)
            k = k.contiguous().view(-1, k.shape[3] * k.shape[4])
            score_map = torch.matmul(q, k)

            b_s, l_s, h_s, w_s = b_one.shape[0], patch_28.shape[1], b_one.shape[2], b_one.shape[3]
            att = F.softmax(score_map * self.softmax_scale, dim=1)
            v = patch_112.contiguous().view(patch_112.shape[0] * patch_112.shape[1], -1)
            attMulV = torch.mm(att, v)

            zi = attMulV.view(b_s, l_s, -1).permute(0, 2, 1)


            # spatial-angular convolutional de-tokenization
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0],
                                          stride=self.stride_1)

            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=paddings[0],
                                                 stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel),
                                                padding=paddings[0],
                                                stride=self.stride_1)

            zi = zi / out_mask

            y = self.W(zi)
            y = b_one + y
            if self.add_SE:
                y_SE = self.SE(y)
                y = self.conv33(torch.cat((y_SE * y, y), dim=1))
            output.append(y)
        output = torch.stack(output, dim=0)
        return output

    def GSmap(self, a, b):
        return torch.matmul(a, b)






class SALSA(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=4, softmax_scale=10, shape=64, p_len=64, in_channels=64
                 , inter_channels=16, use_multiple_size=False, use_topk=False, add_SE=False):
        super(SALSA, self).__init__()
        self.ksize = ksize
        self.shape = shape
        self.p_len = p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size = use_multiple_size
        self.use_topk = use_topk
        self.add_SE = add_SE
        # self.SE=SE_net(in_channels=in_channels)
        self.conv33 = nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                padding=0)
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, b):
        output = []
        for B in range(b.shape[0]):
            b_one = b[B]
            kernel = self.ksize


            # spatial-angular convolutional tokenization
            b1 = self.g(b_one)
            b2 = self.theta(b_one)
            b3 = self.phi(b_one)

            raw_int_bs = list(b1.size())


            patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                          strides=[self.stride_1, self.stride_1],
                                                          rates=[1, 1],
                                                          padding='same')
            patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel,
                                     -1)
            patch_28 = patch_28.permute(0, 4, 1, 2, 3)

            patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                            strides=[self.stride_2, self.stride_2],
                                                            rates=[1, 1],
                                                            padding='same')
            patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel,
                                       -1)
            patch_112 = patch_112.permute(0, 4, 1, 2, 3)

            patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                                strides=[self.stride_2, self.stride_2],
                                                                rates=[1, 1],
                                                                padding='same')
            patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel,
                                           -1)
            patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)

            _, paddings = same_padding(torch.split(b3, 1, dim=0)[0], [self.ksize, self.ksize], [1, 1],
                                       [1, 1])



            # spatial-angular self-attention
            q = patch_28.contiguous().view(patch_28.shape[0] * patch_28.shape[1], -1)
            k = patch_112_2.permute(2, 3, 4, 0, 1)
            k = k.contiguous().view(-1, k.shape[3] * k.shape[4])
            score_map = torch.matmul(q, k)

            b_s, l_s, h_s, w_s = b_one.shape[0], patch_28.shape[1], b_one.shape[2], b_one.shape[3]
            att = F.softmax(score_map * self.softmax_scale, dim=1)
            v = patch_112.contiguous().view(patch_112.shape[0] * patch_112.shape[1], -1)
            attMulV = torch.mm(att, v)

            zi = attMulV.view(b_s, l_s, -1).permute(0, 2, 1)


            # spatial-angular convolutional de-tokenization
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0],
                                          stride=self.stride_1)

            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=paddings[0],
                                                 stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel),
                                                padding=paddings[0],
                                                stride=self.stride_1)

            zi = zi / out_mask

            y = self.W(zi)
            y = b_one + y
            if self.add_SE:
                y_SE = self.SE(y)
                y = self.conv33(torch.cat((y_SE * y, y), dim=1))
            output.append(y)
        output = torch.stack(output, dim=0)
        return output

    def GSmap(self, a, b):
        return torch.matmul(a, b)


class SE_net(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_net, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction, kernel_size=1, stride=1,
                             padding=0)
        self.fc2 = nn.Conv2d(in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, x):
        o1 = self.pool(x)
        o1 = F.relu(self.fc1(o1))
        o1 = self.fc2(o1)
        return o1


class size_selector(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(size_selector, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=intermediate_channels),
            nn.BatchNorm1d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.selector_a = nn.Linear(in_features=intermediate_channels, out_features=out_channels)
        self.selector_b = nn.Linear(in_features=intermediate_channels, out_features=out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        vector = x.mean(-1).mean(-1)
        o1 = self.embedding(vector)
        a = self.selector_a(o1)
        b = self.selector_b(o1)
        v = torch.cat((a, b), dim=1)
        v = self.softmax(v)
        a = v[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b = v[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return a, b



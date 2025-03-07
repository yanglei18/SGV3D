# Copyright (c) Megvii Inc. All rights reserved.
import numpy as np

import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer

from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

from ops.voxel_pooling import voxel_pooling

__all__ = ['BSMLSSFPN']

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class SABlock(nn.Module):
    """ Spatial attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x, y):
        return torch.mul(self.conv(x), self.attention(y))

class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self,  channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.depth2sem = SABlock(channels, channels)
        self.sem2depth = SABlock(channels, channels)

    def forward(self, depth, sem):
        depth_new = depth + self.sem2depth(sem, depth)
        sem_new = sem + self.depth2sem(depth, sem)
        return depth_new, sem_new

class TaskHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, with_head=True):
        super(TaskHead, self).__init__()
        self.with_head = with_head
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.decoder = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1), 
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        if self.with_head:
            self.head = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, return_feat=True):
        if return_feat:
            if self.with_head:
                feat = self.decoder(feat)
                return self.head(feat), feat
            else:
                return feat
        return self.head(self.decoder(feat))

class TaskFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TaskFPN, self).__init__()
        self.reduce_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.self_attention = SABlock(out_channels, out_channels)
    
    def forward(self, feat0, feat1):
        feat0 = self.reduce_conv(F.interpolate(feat0, scale_factor=2, mode='bilinear'))
        feat0_new = feat0 + self.self_attention(feat1, feat0)
        return feat0_new

class MSCThead(nn.Module):
    def __init__(self,
                 in_channels=[512, 512],
                 mid_channels=[512, 256],
                 depth_channels=90,
                 semantic_channels=2,
                 context_channels=80,
                ):
        super(MSCThead, self).__init__()
        # preprocess
        self.reduce_conv0 =  nn.Sequential(
                nn.Conv2d(in_channels[0], mid_channels[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mid_channels[0]), nn.ReLU(inplace=True))
        self.reduce_conv1 =  nn.Sequential(
                nn.Conv2d(in_channels[1], mid_channels[1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mid_channels[1]), nn.ReLU(inplace=True))
        self.bn = nn.BatchNorm1d(27)
        self.scale0_mlp = Mlp(27, mid_channels[0], mid_channels[0])
        self.scale1_mlp = Mlp(27, mid_channels[1], mid_channels[1])
        self.scale0_se = SELayer(mid_channels[0])
        self.scale1_se = SELayer(mid_channels[1])
        self.aspp = ASPP(mid_channels[0], mid_channels[0])
        # stage one
        self.depth_head0= TaskHead(mid_channels[0], mid_channels[0], depth_channels, with_head=False)
        self.semantic_head0 = TaskHead(mid_channels[0], mid_channels[0], semantic_channels)
        self.context_conv0 = nn.Sequential(
            nn.Conv2d(mid_channels[0], mid_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[0]), 
            nn.ReLU(inplace=True)
        ) 
        # combine information
        # self.mtd = MultiTaskDistillationModule(mid_channels[0])
        self.depth_fpn = TaskFPN(mid_channels[0], mid_channels[1])
        self.semantic_fpn = TaskFPN(mid_channels[0], mid_channels[1])
        self.context_fpn = TaskFPN(mid_channels[0], mid_channels[1])
        # stage two
        self.depth_head1 = TaskHead(mid_channels[1], mid_channels[1], depth_channels)
        self.semantic_head1 = TaskHead(mid_channels[1], mid_channels[1], semantic_channels)
        self.context_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels[1], mid_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[1]), 
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels[1], context_channels, kernel_size=1, stride=1, padding=0)
        ) 
    
    @autocast(False)
    def forward(self, x, mats_dict):
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4, 4).repeat(1, 1, num_cams, 1, 1)
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))

        # preprocess
        B, N, C, H, W = x[0].shape
        scale0_feat = x[0].view(B * N, C, H, W).float()
        B, N, C, H, W = x[1].shape
        scale1_feat = x[1].view(B * N, C, H, W).float()
        scale0_feat = self.reduce_conv0(scale0_feat)
        scale1_feat = self.reduce_conv1(scale1_feat)
        scale0_se = self.scale0_mlp(mlp_input)[..., None, None]
        scale1_se = self.scale1_mlp(mlp_input)[..., None, None]
        scale0_feat = self.scale0_se(scale0_feat, scale0_se)
        scale1_feat = self.scale1_se(scale1_feat, scale1_se)
        scale0_feat = self.aspp(scale0_feat)
        # stage one
        depth_feat = self.depth_head0(scale0_feat)
        semantic0, semantic_feat = self.semantic_head0(scale0_feat)
        context_feat = self.context_conv0(scale0_feat)
        # combine information
        # depth_feat, semantic_feat = self.mtd(depth_feat, semantic_feat)
        depth_feat = self.depth_fpn(depth_feat, scale1_feat)
        semantic_feat = self.semantic_fpn(semantic_feat, scale1_feat)
        context_feat = self.context_fpn(context_feat, scale1_feat)
        # stage two
        depth1 = self.depth_head1(depth_feat, return_feat=False)
        semantic1 = self.semantic_head1(semantic_feat, return_feat=False)
        context1 = self.context_conv1(context_feat)
        return (depth1, semantic1, context1, semantic0)

class BSMLSSFPN(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim, output_channels, downsample_factor,  img_backbone_conf,
                 img_neck_conf, height_net_conf, is_train_height, is_bsm):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            height_net_conf (dict): Config for height net.
        """

        super(BSMLSSFPN, self).__init__()
        self.downsample_factor = downsample_factor // 2
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels
        self.is_train_height = is_train_height

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())
        self.height_channels, _, _, _ = self.frustum.shape
        
        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_backbone.init_weights()
        self.img_neck_16 = build_neck(img_neck_conf)
        self.img_neck_16.init_weights()
        
        img_neck_conf['upsample_strides'] = [0.5, 1, 2, 4]
        self.img_neck_8 = build_neck(img_neck_conf)
        self.img_neck_8.init_weights()
        self.height_net = self._configure_height_net(height_net_conf)
        

    def _configure_height_net(self, height_net_conf):
        return MSCThead(
            in_channels=height_net_conf['in_channels'],
            mid_channels=height_net_conf['mid_channels'],
            depth_channels=self.height_channels,
            semantic_channels=height_net_conf['semantic_channels'],
            context_channels=self.output_channels,
        )

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        
        # DID
        alpha = 1.5
        d_coords = np.arange(self.d_bound[2]) / self.d_bound[2]
        d_coords = np.power(d_coords, alpha)
        d_coords = self.d_bound[0] + d_coords * (self.d_bound[1] - self.d_bound[0])
        d_coords = torch.tensor(d_coords, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum
    
    def height2localtion(self, points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        reference_heights = reference_heights.view(batch_size, num_cams, 1, 1, 1, 1,
                                                   1).repeat(1, 1, points.shape[2], points.shape[3], points.shape[4], 1, 1)
        height = -1 * points[:, :, :, :, :, 2, :] + reference_heights[:, :, :, :, :, 0, :]
        
        points_const = points.clone()
        points_const[:, :, :, :, :, 2, :] = 10
        points_const = torch.cat(
            (points_const[:, :, :, :, :, :2] * points_const[:, :, :, :, :, 2:3],
             points_const[:, :, :, :, :, 2:]), 5)
        combine_virtual = sensor2virtual_mat.matmul(torch.inverse(intrin_mat))
        points_virtual = combine_virtual.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points_const)
        ratio = height[:, :, :, :, :, 0] / points_virtual[:, :, :, :, :, 1, 0]
        ratio = ratio.view(batch_size, num_cams, ratio.shape[2], ratio.shape[3], ratio.shape[4], 1, 1).repeat(1, 1, 1, 1, 1, 4, 1)
        points = points_virtual * ratio
        points[:, :, :, :, :, 3, :] = 1
        combine_ego = sensor2ego_mat.matmul(torch.inverse(sensor2virtual_mat))
        points = combine_ego.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        return points
    
    def get_geometry(self, sensor2ego_mat, sensor2virtual_mat, intrin_mat, ida_mat, reference_heights, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        points = self.height2localtion(points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights) 
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)

            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        img_feats = self.img_backbone(imgs)
        img_feats0 = self.img_neck_16(img_feats)[0]
        img_feats1 = self.img_neck_8(img_feats)[0]
        img_feats0 = img_feats0.reshape(batch_size * num_sweeps, num_cams,
                                        img_feats0.shape[1], img_feats0.shape[2],
                                        img_feats0.shape[3])
        img_feats1 = img_feats1.reshape(batch_size * num_sweeps, num_cams,
                                        img_feats1.shape[1], img_feats1.shape[2],
                                        img_feats1.shape[3])
        return [img_feats0, img_feats1]

    def _forward_height_net(self, feat, mats_dict):
        return self.height_net(feat, mats_dict)

    def _forward_voxel_net(self, img_feat_with_height):
        return img_feat_with_height

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_height (bool, optional): Whether to return height.
                Default: False.
        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        '''
        source_features = img_feats[:, 0, ...]
        source_features = source_features.reshape(batch_size * num_cams,
                                                  source_features.shape[2],
                                                  source_features.shape[3],
                                                  source_features.shape[4])
        '''
        out = self._forward_height_net(img_feats, mats_dict) # depth1, semantic1, context1, semantic0
        height = out[0].softmax(dim=1)
        semantic = out[1].softmax(dim=1)
        tran_feat = out[2]
        tran_feat = torch.cat((tran_feat, semantic), dim=1)

        mask = semantic[:, 0, :, :].unsqueeze(1) > 0.45  # background
        tran_feat = tran_feat * (1 - mask.int())

        img_feat_with_height = height.unsqueeze(1) * tran_feat.unsqueeze(2)
        img_feat_with_height = self._forward_voxel_net(img_feat_with_height)

        img_feat_with_height = img_feat_with_height.reshape(
            batch_size,
            num_cams,
            img_feat_with_height.shape[1],
            img_feat_with_height.shape[2],
            img_feat_with_height.shape[3],
            img_feat_with_height.shape[4],
        )
        
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['sensor2virtual_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict['reference_heights'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        img_feat_with_height = img_feat_with_height.permute(0, 1, 3, 4, 5, 2)
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int() 
        feature_map = voxel_pooling(geom_xyz, img_feat_with_height.contiguous(),
                                   self.voxel_num.cuda())
        
        if self.is_train_height:
            return feature_map.contiguous(), (out[3], out[1]) # semantic0, semantic1 
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict)
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if self.is_train_height else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict)
                ret_feature_list.append(feature_map)

        if self.is_train_height:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)

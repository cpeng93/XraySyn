import torch
import torch.nn as nn
import pickle
from ..networks.unet import UnetGenerator
from ..networks.rdn_meta import make_model
from ..utils.torch import print_model, NormLayer
from .base import Base
from ..networks.common import NLayerDiscriminator, GANLoss
from ..utils.geometry import get_6dofs_transformation_matrix
import numpy as np
from ..networks.drr_projector_new import DRRProjector

class XraySynModel(Base):
    def __init__(self, num_feats_3d=16, num_layers_3d=2, num_feats_2d=32,
        num_layers_2d=5, volume_shape=(160, 192, 192), detector_shape=(160, 192),
        pixel_size=(1, 1), interp="nearest", lr=1e-4, beta1=0.5,
        learn={"vol_w": 0.0, "vol_t": "l1", "proj1_w": 0.0, "proj2_w": 1.0,
        "proj_t": "l1", "proj_adv": 0.0}, device="cuda:0"):
        super(XraySynModel, self).__init__(device, ["net3d","net2d","netD"])

        self.netD = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3,
                norm_layer="instance").to(self.device)

        self.net3d = UnetGenerator(1, 3, num_downs=6,
            dimension="3d", ngf=96, norm_layer="batch",up_layer="upsample3D").to(self.device)
        self.net2d = make_model().to(self.device)
        pixel = 1
        self.proj = DRRProjector(
            mode="forward", volume_shape=(128,128,128), detector_shape=(128,128),
            pixel_size=(pixel, pixel), interp=interp, source_to_detector_distance=1200).to(self.device)
        
        self.backproj = DRRProjector(
            mode="backward", volume_shape=(128,128,128), detector_shape=(128,128),
            pixel_size=(pixel, pixel), interp=interp, source_to_detector_distance=1200).to(self.device)

        self.norm = NormLayer()
        self.learn = learn
        self.obj_proj = nn.L1Loss()
        self.obj_vol = nn.L1Loss()
        self.obj_bce = nn.BCELoss()
        self.avgpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bone_absorb = pickle.load(
            open('simplified_bone_absorb_2d.pt', 'rb')).to(self.device)
        self.tissue_absorb = pickle.load(
            open('simplified_tissue_absorb_2d.pt', 'rb')).to(self.device)
        self.l2loss = nn.MSELoss()
        self.views = []

        for i in range(21):
            self.views.append([10, i])


        self.net3d = nn.DataParallel(self.net3d)
        self.net2d = nn.DataParallel(self.net2d)
        self.netD = nn.DataParallel(self.netD)
        self.proj = nn.DataParallel(self.proj)
        self.backproj = nn.DataParallel(self.backproj)
        self.bone_absorb = nn.DataParallel(self.bone_absorb)
        self.tissue_absorb = nn.DataParallel(self.tissue_absorb)
        self.norm = nn.DataParallel(self.norm)
        for param in self.net3d.parameters():
            param.requires_grad = False
        for param in self.bone_absorb.parameters():
            param.requires_grad = False
        for param in self.tissue_absorb.parameters():
            param.requires_grad = False

        self.optimG = torch.optim.Adam(
            self.net2d.parameters(),
            lr=lr, betas=(beta1, 0.9))

        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=lr, betas=(beta1, 0.9))        


        self.obj_gan_p = GANLoss(gan_mode='lsgan').to(self.device)        
        print_model(self.net3d)
        print_model(self.net2d)

    def get_T(self, inp):
        param =np.asarray(inp)
        param = param * np.pi
        T = get_6dofs_transformation_matrix(param[3:], param[:3])
        T = torch.FloatTensor(T[np.newaxis, ...]).to(self.device)
        return torch.cat([T,T,T,T])



    def ct2xray(self, vol, bone,T_in):
        vol = 0.0008088*(vol*5000-1000)+1.030
        bone_vol = vol * bone
        tissue_vol = vol * (1-bone)
        bone_proj = self.proj(bone_vol, T_in)
        tissue_proj = self.proj(tissue_vol, T_in)
        atten_bone = self.bone_absorb(bone_proj)
        atten_tissue = self.tissue_absorb(tissue_proj)
        atten_proj = atten_bone+atten_tissue
        out_new = torch.exp(atten_proj).sum(dim=1).view(vol.shape[0],1,128,128)
        out_new = self.norm(out_new.max() - out_new)
        return out_new, torch.cat([bone_proj, tissue_proj],1)


    def mat2xray(self, mat,fill=False):
        if fill:
            atten_bone = self.tissue_absorb(mat[:,[0]])
        else:
            atten_bone = self.bone_absorb(mat[:,[0]])
        atten_tissue = self.tissue_absorb(mat[:,[1]])
        atten_proj = atten_bone+atten_tissue
        out_new = torch.exp(atten_proj).sum(dim=1).view(mat.shape[0],1,256,256)
        out_new = self.norm(out_new.max() - out_new)
        return out_new

    def set_input(self, xray, T_in, T_other):
        with torch.no_grad():
            self.xray = xray
            self.T_in = T_in
            self.T_other = T_other

    def test(self, xray):
        T_in = self.get_T([1,0,0,0,0,0])
        xray128 = self.avgpool(xray)
        with torch.no_grad():
            vol_in = self.backproj(xray128, T_in)
            vol_pred_temp = self.net3d(vol_in)*0.5+0.5
            bone_mask = vol_pred_temp[:,[0]]
            bone_ct = vol_pred_temp[:,[1]]*bone_mask
            tissue_ct = vol_pred_temp[:,[2]]*(1-bone_mask)
            vol_pred = bone_ct + tissue_ct
            xray_other_pred = []
            mat_other_pred = []
            xray_other_refine = []
            mat_other_refine = []
            for view in self.views:
                print([-0.05+0.1*view[0]/20+1,-0.05+0.1*view[1]/20,0,0,0,0])
                T_other = self.get_T([-0.05+0.1*view[0]/20+1,-0.05+0.1*view[1]/20,0,0,0,0])
                xray2_pred, mat2_pred = self.ct2xray(vol_pred, bone_mask, T_other)
                mat2_refine = self.net2d(mat2_pred,xray) + self.upsample(mat2_pred)
                xray2_refine = self.mat2xray(mat2_refine)
                xray_other_pred.append(self.upsample(xray2_pred))
                mat_other_pred.append(self.upsample(mat2_pred))
                xray_other_refine.append(xray2_refine)
                mat_other_refine.append(mat2_refine)
        return xray_other_pred, xray_other_refine, mat_other_pred, mat_other_refine, vol_pred, bone_mask

    def optimize(self):
        with torch.no_grad():
            xray = self.avgpool(self.xray)
            vol_in = self.backproj(xray, self.T_in)
            vol_pred_temp = self.net3d(vol_in)*0.5+0.5
            bone_mask = vol_pred_temp[:,[0]]
            bone_ct = vol_pred_temp[:,[1]]*bone_mask
            tissue_ct = vol_pred_temp[:,[2]]*(1-bone_mask)
            vol_pred = bone_ct + tissue_ct
            xray_pred, mat_pred = self.ct2xray(vol_pred, bone_mask, self.T_in)
            xray2_pred, mat2_pred = self.ct2xray(vol_pred, bone_mask, self.T_other)

        mat2_res = self.net2d(mat2_pred, self.xray)
        mat_res = self.net2d(mat_pred, self.xray)
        mat2_refine = mat2_res + self.upsample(mat2_pred)
        mat_refine = mat_res + self.upsample(mat_pred)
        xray2_refine = self.mat2xray(mat2_refine)
        xray_refine = self.mat2xray(mat_refine)
        xray2_fill = self.mat2xray(mat2_refine,fill=True)
        xray_fill = self.mat2xray(mat_refine,fill=True)

        self.optimD.zero_grad()
        real = self.netD(self.xray)
        fake2 = self.netD(xray2_refine.detach())
        error_fake = self.obj_gan_p(fake2, False)
        error_real = self.obj_gan_p(real, True)
        loss_D_p = error_fake + error_real
        loss_D_p.backward()
        self.optimD.step()
        self._record_loss("GAN_D", loss_D_p)

        fake2 = self.netD(xray2_refine)
        loss_gan = self.obj_gan_p(fake2, True)
        self._record_loss("GAN_G", loss_gan)


        loss_l1 = self.obj_proj(xray_refine, self.xray)
        loss_sparse = 0.5*self.obj_proj(self.avgpool(mat_refine[:,0]), mat_pred[:,0]) + 0.5*self.obj_proj(self.avgpool(mat2_refine[:,0]), mat2_pred[:,0])
        if loss_sparse.item() > 1:
            loss = loss_l1 + 0.02*loss_gan + 0.005*loss_sparse
            self._record_loss(f"SparseYes", loss_sparse)
        else:
            loss = loss_l1 + 0.02*loss_gan
            self._record_loss(f"SparseNo", loss_sparse)

        self._record_visual("xraygt", xray[:, 0])
        self._record_visual("xray1pred", self.upsample(xray_pred)[:, 0])
        self._record_visual("xray1refine", xray_refine[:, 0])
        self._record_visual("xray1fill", xray_fill[:, 0])
        self._record_visual("xray1predbone", self.upsample(mat_pred)[:, 0])
        self._record_visual("xray1predtissue", self.upsample(mat_pred)[:, 1])
        self._record_visual("xray1resbone", mat_res[:, 0])
        self._record_visual("xray1restissue", mat_res[:, 1])
        self._record_visual("xray1refinebone", mat_refine[:, 0])
        self._record_visual("xray1refinetissue", mat_refine[:, 1])
        self._record_visual("xray2pred", self.upsample(xray2_pred)[:, 0])
        self._record_visual("xray2refine", xray2_refine[:, 0])
        self._record_visual("xray2fill", xray2_fill[:, 0])
        self._record_visual("xray2predbone", self.upsample(mat2_pred)[:, 0])
        self._record_visual("xray2predtissue", self.upsample(mat2_pred)[:, 1])
        self._record_visual("xray2refinebone", mat2_refine[:, 0])
        self._record_visual("xray2refinetissue", mat2_refine[:, 1])
        self._record_loss(f"L1", loss_l1)
        self._record_loss(f"GAN_G", loss_gan)
        self.optimG.zero_grad()
        loss.backward()
        self.optimG.step()

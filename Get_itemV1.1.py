# import numpy
import torch

import datetime
import pdb
import sys
import torch
import torch.nn as nn
import torchvision.utils as tvis
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import matplotlib as mplot
import random
import os
#---import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
#import matplotlib.pyplot as plt
import random
import os
#import matplotlib.pyplot as plt
# from PIL import Image

# try:
#     from PIL import __version__
# except ImportError:
#     from PIL import PILLOW_VERSION as __version__

import torch.utils.data as data
import pdb
import datetime
from PIL import Image
import numpy as np
import scipy.spatial.distance




path_input="/storage/3050/MazurM/db/7classes_masks_croppedv1/Input/"
path_masks="/storage/3050/MazurM/db/7classes_masks_croppedv1/Masks/"
num_sources=7

#--------------HR net-------------------------------------------------------------



#-----------------creating masks from noised masks, then cropping images and masks from original noised sources-----------


#realisation of code below (reading from sources) depends on structure and hierarchy of images and masks that are being used as sources for get_item function
# we use input images from  different "sources" (like Google and Yandex maps,but only one mask all the time), so we handle the probability of using the certain source input by probability (MARKED  #1)


path_soruces_learn="/storage/3050/MazurM/db/OrigNoiseSegmentBulat/learn/"  # masks and input images are storing in the same directory
folder_names=[]

names_img_inputs_1=["bing_sat","esri_sat","google_sat","mapbox_sat_with_api"] # 1 frist layer in input (satellite photo)
extension_im_inputs_1="png" # for ALL images!!!
names_img_inputs_2=["another_osm__maps","esri_map","google_maps","cartodb"] # 2 second layer in input (masks from Google,Yandex etc.)
extension_im_inputs_2="png" # for ALL images!!!

name_mask_label="LABEL_5" # all names for mask are the SAME !!!
extension_name_mask_label="png" # for ALL mask labels!!!

all_names=os.listdir(path_soruces_learn)
for name in all_names:
    if(os.path.isdir(path_soruces_learn+name)):
        folder_names.append(name)


def img_crop(input_img_1,input_img_2,mask_img,width_new,height_new): # defined for two sources images
    width_big,height_big,channels=input_img_1.shape
    width_point=random.randint(0,width_big-width_new)
    height_point=random.randint(0,height_big-height_new)

    input_img_crp_1=input_img_1[width_point:width_point+width_new,height_point:height_point+height_new,0:3]
    input_img_crp_2=input_img_2[width_point:width_point+width_new,height_point:height_point+height_new,0:3]
    mask_img_crp=mask_img[width_point:width_point+width_new,height_point:height_point+height_new,0:3]

    return input_img_crp_1, input_img_crp_2,mask_img_crp
    pass

#  list_of_colors is a list of numpy arrays; the mask and item in list of colours should have the same dimension of channels, third dimension: Width, Height,channels

def cosine_dist(vec1,vec2):
    # vec1.astype(np.uint64)
    # vec2.astype(np.uint64)
    #-----------------------------------
    dims, *_=vec1.shape
    vecmult1=0.0
    vecmult2=0.0
    vecmult12=0.0
    for d in range(dims):
        wtf1=vec1[d]
        vecmult1+=vec1[d]*vec1[d]
        # vecmult1+=wtf1*wtf1
        # vecmult1+=9999999
        vecmult2+=vec2[d]*vec2[d]
        vecmult12+=vec1[d]*vec2[d]
    numerator=vecmult12
    denumerator=(vecmult1**0.5)*(vecmult2**0.5)
    numerator=np.sum(vec1*vec2)
    wtf1=vec1*vec1
    wtf11=np.dot(vec1,vec1)
    #-----------------------------------
    vec1_len=np.sum(vec1*vec1)
    vec2_len=np.sum(vec2*vec2)
    denumerator=(vec1_len**0.5) * (vec2_len**0.5)
    print(numerator)
    print(denumerator)
    print(numerator/denumerator)
    print(1-numerator/denumerator)
    print(scipy.spatial.distance.cosine(vec1,vec2))
    return numerator/denumerator

def mask_creator(mask,list_of_colors): # list_of_colors -is a numpy array, first dimension = number of classes, second dimension represents number of channels; MASK's type is numpy array
    width,height,channels, *_=mask.shape
    classes_amount,*_=list_of_colors.shape
    mask_crp_clean=np.zeros((width,height,channels),dtype=np.uint64) # ,dtype=np.uint64
    for w in range(width):
        for h in range(height):
            current_pix=mask[w,h,:]
            similarity=1.0
            color_index=0
            for iter in range(classes_amount):
                vec1=mask[w,h,0:3]
                vec1=vec1.astype(np.float64)
                vec2=list_of_colors[iter,0:3]
                vec2=vec1.astype(np.float64)
                # tmp=scipy.spatial.distance.cosine(mask[w,h,0:3],list_of_colors[iter,0:3]) # 1-cosine distance # wtf? return -241.0034500
                tmp=cosine_dist(mask[w,h,0:3],list_of_colors[iter,0:3]) #<<<<<<<<<<<<<<<<
                if(similarity<tmp):
                    similarity=tmp
                    color_index=iter
            mask_crp_clean[w,h]=list_of_colors[color_index]

    pass


def get_item(item,pixels_classes):
    width_new=256
    height_new=256
    path_soruces_learn = "/storage/3050/MazurM/db/OrigNoiseSegmentBulat/Temp/"  # masks and input images are storing in the same directory
    folder_names = []

    names_img_inputs_1 = ["bing_sat", "esri_sat", "google_sat","mapbox_sat_with_api"]  # 1 frist layer in input (satellite photo)
    extension_im_inputs_1 = "png"  # for ALL images!!!
    names_img_inputs_2 = ["another_osm__maps", "esri_map", "google_maps",
                          "cartodb"]  # 2 second layer in input (masks from Google,Yandex etc.)
    extension_im_inputs_2 = "png"  # for ALL images!!!

    name_mask_label = "LABEL_5"  # all names for mask are the SAME !!!
    extension_name_mask_label = "png"  # for ALL mask labels!!!

    for name in all_names:
        if (os.path.isdir(path_soruces_learn + name)):
            folder_names.append(name)
    in_i_1=random.randint(0,len(names_img_inputs_1)-1)
    in_i_2=random.randint(0,len(names_img_inputs_2)-1)

    wtf1=path_soruces_learn+folder_names[item]
    wtf2="/"+names_img_inputs_1[in_i_1] # PROBLEM
    wtf3="."+extension_im_inputs_1


    # wtf1=path_soruces_learn+folder_names+"/"+str(names_img_inputs_1[in_i_1])+"."+extension_im_inputs_1
    print("debug")
    with Image.open(path_soruces_learn+folder_names[item]+"/"+names_img_inputs_1[in_i_1]+"."+extension_im_inputs_1) as input_img_1, \
            Image.open(path_soruces_learn+folder_names[item]+"/"+names_img_inputs_2[in_i_2]+"."+extension_im_inputs_2) as input_img_2, \
            Image.open(path_soruces_learn+folder_names[item]+"/"+name_mask_label+"."+extension_name_mask_label) as mask_img_nosiy:
        input_img_1 =np.asarray(input_img_1)
        input_img_2 =np.asarray(input_img_2)
        mask_img_nosiy =np.asarray(mask_img_nosiy,dtype=np.uint64)
        input_img_crp_1,input_img_crp_2,mask_img_crp=img_crop(input_img_1,input_img_2,mask_img_nosiy,width_new,height_new)
        mask_crp_clean=mask_creator(mask_img_crp,pixels_classes)
        print("debug1")


        # plt.imshow(input_img_crp_1)
        # plt.show()
        # plt.imshow(input_img_crp_2)
        # plt.show()
        # plt.imshow(mask_img_crp)
        # plt.show()


pixels_classes = np.array(
    [[255, 255, 255, 255],
     [250, 10, 222, 255],
     [0, 255, 0, 255],
     [255, 0, 0, 255],
     [50, 0, 0, 255],
     [30, 0, 0, 255],
     [0, 0, 255, 255]],dtype=np.uint64) # ,dtype=np.uint64


sizes=pixels_classes.shape
print(pixels_classes.shape)
print(sizes[0])
print(sizes[1])
print("debug 1")


#[item] in range of len(folder_names)
# temporary it value is 0-3
get_item(3,pixels_classes)


print("debug 1")

# for i in range (len(names))



# print [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]

#-----------------creating masks from noised masks, then cropping images and masks from original noised sources-----------

#
# class Dataset_custom(data.Dataset): # dataset creator for GAN, paired dataset (images)
#     def __init__(self): #def __init__(self, str): str - path or TYPE of forward
#         self.path_inputs = path_input
#         self.path_masks = path_masks
#         self.names = os.listdir(path_masks)
#         print("--->>>TRAIN SET HAS BEEN CHOOSEN<<<------")
#         # if((self.typer=="train") or (self.typer!=0)):
#         #     self.path_in=path_learn_in
#         #     self.path_targets=path_learn_targets
#         #     self.names=os.listdir(path_learn_in)
#         #     print("--->>>TRAIN SET HAS BEEN CHOOSEN<<<------")
#         #
#         # else:
#         #     print("--->>>VALIDATION SET HAS BEEN CHOOSEN<<<------")
#         #     self.path_in=path_vaild_in
#         #     self.path_targets=path_vaild_targets
#         #     self.names=os.listdir(path_vaild_in)
#     def __getitem__(self, item):
#         self.num_sources=num_sources
#         height=256
#         width=256
#         input=np.array(Image.open(self.path_inputs+self.names[item][0:-8]+"input"+"0.png"))[:,:,0:3]
#         for i in range(1,self.num_sources):
#             tmp=np.array(Image.open(self.path_inputs+self.names[item][0:-8]+"input"+str(i)+".png"))[:,:,0:3]
#             input=np.concatenate((input,tmp),axis=2)
#
#
#         pixels_classes = np.array(
#             [[255, 255, 255, 255],
#              [250, 10, 222, 255],
#              [0, 255, 0, 255],
#              [255, 0, 0, 255],
#              [50, 0, 0, 255],
#              [30, 0, 0, 255],
#              [0, 0, 255, 255]])
#         mask_img = np.array(Image.open(self.path_masks + self.names[item]))
#         # width, height,_=mask_img.size() # TypeError: 'int' object is not callable
#         # mask = np.zeros((width, height, 1), dtype=np.uint8) # was a problem with RuntimeError: 1only batches of spatial targets supported (non-empty 3D tensors) but got targets of size: : [4, 256, 256, 1]
#         mask = np.zeros((width, height), dtype=np.uint8)
#         for w in range(width):
#             for h in range(height):
#                 mask[w, h] = np.where((pixels_classes == mask_img[w, h]).all(axis=1))[0]
#         input=torch.from_numpy(input)
#         input=input.type(torch.float32)/255.0
#         input=input.permute(2,0,1)
#         mask=torch.from_numpy(mask)
#         #----creating mask using image realB -------[width,height,1] finished
#
#         # #----creating mask using image realB ------- [width,height,num classes]
#         # # num_classes=7
#         # pixels_classes = np.array(
#         #     [[255, 255, 255, 255],
#         #      [250, 10, 222, 255],
#         #      [0, 255, 0, 255],
#         #      [255, 0, 0, 255],
#         #      [50, 0, 0, 255],
#         #      [30, 0, 0, 255],
#         #      [0, 0, 255, 255]])
#         # pixels_classes=pixels_classes[:,0:3]
#         # num_classes,_=pixels_classes.shape
#         # pixels = pixels_classes[:, 0:3]
#         # maskB=np.zeros((width,height,num_classes))
#         # for w in range(width):
#         #     for h in range(height):
#         #         wtf1 = realB_img[w, h]
#         #         class_index = np.where((pixels_classes == realB_img[w, h]).all(axis=1))[0]
#         #         # print("class_index=",class_index)
#         #         # print("img_mask[w,h]= ",img_mask[w,h])
#         #         realB_img[w, h, class_index] = 1
#         # #----creating mask using image realB -------
#
#
#
#         # realB_img=realB_img.type(torch.float32)/255.0
#         # realB_img.requires_grad=False
#         # number of classes= 7
#         # dictionary of classes
#
#
#         mask=mask.type(torch.int64)
#         return  input,mask #label is synonim of target here
#
#     def __len__(self):
#         return len(self.names)
#
#
# # class GANmodel(nn.Module):
# #     def __init__(self):
# #         super(GANmodel,self).__init__()
# #         #---------------------Generator-------------------------
# #         self.input_ch_g=1 # because of noize!!!
# #         self.output_ch=3  # depends on PNG or JPG format, but keep it in mind, now i use 3 channels because i slice the fourth
# #         self.ngf_g=64 # number of min filters
# #         self.conv2d11=nn.Conv2d(in_channels=self.input_ch_g,out_channels=self.output_ch,kernel_size=3,stride=1,padding=1)
# #         self.activ=nn.LeakyReLU()
# #         self.Generator=nn.Sequential(self.conv2d11,self.activ) # Generator
# #         #---------------------Discriminator---------------------
# #         self.input_ch=3
# #         self.ngf=32
# #         self.dense11=nn.Conv2d(in_channels=self.input_ch,out_channels=self.output_ch,stride=1,kernel_size=1,padding=0)
# #         self.activ=nn.LeakyReLU()
# #         self.activTanh=nn.Tanh()
# #         self.midBatchNorm=nn.BatchNorm2d(self.ngf)
# #         self.endBatchNorm=nn.BatchNorm2d(self.ngf+self.output_ch)
# #         self.Discriminator=nn.Sequential(self.dense11,self.activ) # Discriminator
# #     def forward(self,noise, realA): # noise, realA, //input[0] , input[1]
# #         fakeA=self.Generator(noise)
# #         print("fakeA.size()=",fakeA.size())
# #         print("realA.size()=",realA.size())
# #         realA=realA.permute(0,3,1,2)
# #         print("realA.size()=",realA.size())
# #         DoutFake=self.Discriminator(fakeA.detach())
# #         DoutReal=self.Discriminator(realA)
# #         return fakeA,DoutFake,DoutReal
# #         # #------------------------------------
# #         # fakeA=self.Generator(noise)
# #         # print(fakeA.size())
# #         # print(realA.size())
# #         # realA=realA.permute(0,3,1,2)
# #         # print(realA.size())
# #         # DoutFake=self.Discriminator(fakeA.detach())
# #         # DoutReal=self.Discriminator(realA.permute(0,3,1,2))
# #         # return fakeA,DoutFake,DoutReal
# #         # #------------------------------------
# #
# #         # #-------------Generator-----------
# #         # out=self.conv2d11(input)
# #         # out=self.midBatchNorm(out)
# #         # out=self.activ(out)
# #         # side=out
# #         # print("side size=",side.size())
# #         # out=self.conv2d12(input) # <<<<<---------------- problem RuntimeError: Given groups=1, weight of size 64 1 9 9, expected input[4, 64, 256, 256] to have 1 channels, but got 64 channels instead
# #         # out=self.midBatchNorm(out)
# #         # out=self.activ(out)
# #         # print("pre conv size=",out.size())
# #         # out=self.deconv11(out)
# #         # print("after conv size=",out.size())
# #         # out=self.activ(out)
# #         # out=self.midBatchNorm(out)
# #         # out=torch.cat([out,side],dim=1)  #<<<<<--------Sizes of tensors must match except in dimension 1. Got 250 and 256 in dimension 2
# #         # out=self.conv2dend(out)
# #         # out=self.outBatchNorm(out)
# #         # fakeA=self.activ(out)
# #         # #---------------------Discriminator---------------------
# #         # out=self.dense11(input)
# #         # out=self.activ(out)
# #         # side=self.conv12(input)
# #         # side=self.midBatchNorm(side)
# #         # side=self.activ(side)
# #         # out=torch.cat([out,side],dim=1)
# #         # out=self.conv21(out)
# #         # out=self.activ(out)
# #         # out=self.endBatchNorm(out)
# #         # out=self.conv22(out)
# #         # out=self.activ(out)
# #         # out=self.convEnd(out)
# #         # out=self.activTanh(out)
# #         # return out
#
# class inceptionModule(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(inceptionModule,self).__init__()
#         self.num_sources=num_sources
#         self.conv113=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
#         self.conv512=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=5,stride=1,padding=2)
#         self.conv137=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,stride=1,padding=3)
#         self.conv101=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
#         self.BN=nn.BatchNorm2d(out_channels)
#         self.activ=nn.LeakyReLU()
#     def forward_inception(self,input):
#         part1=self.activ(self.conv113(input))
#         part2=self.activ(self.conv512(input))
#         part3=self.activ(self.conv137(input))
#         part4=self.activ(self.conv101(input))
#         out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#
# class HRnet(nn.Module):
#     def __init__(self,num_sources,ch_per_soruce=3):
#
#         # SPK for conv, SPDK_OP for TransConv(order of arguments)
#         # self.first_layer_in_channels=first_layer_in_channels
#         # self.layer_out_channels=layer_out_channels
#         self.out_ch_line_1=64
#         self.out_ch_line_2=128
#         self.out_ch_line_3=256
#         self.out_ch_line_4=512
#         self.num_classes=7 # equal the number of classes
#         self.num_sources=num_sources
#         self.ch_per_soruce=ch_per_soruce
#         super(HRnet,self).__init__()
#         self.input_ch=1 # because of noize!!!
#         self.output_ch=3  # depends on PNG or JPG format, but keep it in mind
#         self.ngf=64 # number of min filters
#         self.conv2d256to126=nn.Conv2d(in_channels=self.input_ch,out_channels=self.ngf,kernel_size=3,stride=1,padding=1)
#         self.conv2d12=nn.Conv2d(in_channels=self.input_ch,out_channels=self.ngf,kernel_size=9,stride=3,padding=4)
#         self.deconv11=nn.ConvTranspose2d(in_channels=self.ngf,out_channels=self.ngf,kernel_size=5,dilation=1,padding=2,stride=3)
#         self.conv2dend=nn.Conv2d(in_channels=2*self.ngf,out_channels=self.output_ch,kernel_size=3,stride=1,padding=1)
#         self.activ=nn.LeakyReLU()
#         # self.inBatchNorm=nn.BatchNorm2d(self.input_ch)
#         self.midBatchNorm=nn.BatchNorm2d(self.ngf)
#         self.outBatchNorm=nn.BatchNorm2d(self.output_ch)
#         #--------------------------------LINE 1 Inception Modules-----------------------------------
#             #-----------------------------inception module--------1 line--1 module--------
#         self.conv113_l11=nn.Conv2d(in_channels=self.ch_per_soruce*self.num_sources,out_channels=self.num_sources*self.ch_per_soruce*self.num_classes,kernel_size=3,stride=1,padding=1) # int-> 7*3=21 channels, out-> 7*3*7=147 channels
#         self.conv512_l11=nn.Conv2d(in_channels=self.ch_per_soruce*self.num_sources,out_channels=self.num_sources*self.ch_per_soruce*self.num_classes,kernel_size=5,stride=1,padding=2)
#         self.conv137_l11=nn.Conv2d(in_channels=self.ch_per_soruce*self.num_sources,out_channels=self.num_sources*self.ch_per_soruce*self.num_classes,kernel_size=7,stride=1,padding=3)
#         self.conv101_l11=nn.Conv2d(in_channels=self.ch_per_soruce*self.num_sources,out_channels=self.num_sources*self.ch_per_soruce*self.num_classes,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l11=nn.Conv2d(in_channels=self.num_sources*self.ch_per_soruce*self.num_classes*4,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#         self.BN_l1=nn.BatchNorm2d(self.out_ch_line_1)
#         self.activ=nn.LeakyReLU()
#             #-----------------------------inception module--------1 line--1 module--------
#             #-----------------------------inception module--------1 line--2 part modules--------
#         self.conv113_l12=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,kernel_size=3,stride=1,padding=1) # int-> 7*3=21 channels, out-> 7*3*7=147 channels
#         self.conv512_l12=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,kernel_size=5,stride=1,padding=2)
#         self.conv137_l12=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,kernel_size=7,stride=1,padding=3)
#         self.conv101_l12=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l12=nn.Conv2d(in_channels=self.out_ch_line_1*4,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------1 line--2 part modules--------
#             #-----------------------------inception module--------1 line--4 part modules--------
#         self.conv113_l14=nn.Conv2d(in_channels=self.out_ch_line_1*4,out_channels=self.out_ch_line_1,kernel_size=3,stride=1,padding=1) # in channels= current line channels * amount of all lines
#         self.conv512_l14=nn.Conv2d(in_channels=self.out_ch_line_1*4,out_channels=self.out_ch_line_1,kernel_size=5,stride=1,padding=2)
#         self.conv137_l14=nn.Conv2d(in_channels=self.out_ch_line_1*4,out_channels=self.out_ch_line_1,kernel_size=7,stride=1,padding=3)
#         self.conv101_l14=nn.Conv2d(in_channels=self.out_ch_line_1*4,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l14=nn.Conv2d(in_channels=self.out_ch_line_1*4,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#         # self.conv113_l14=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_1,kernel_size=3,stride=1,padding=1) # int-> 7*3=21 channels, out-> 7*3*7=147 channels
#         # self.conv512_l14=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_1,kernel_size=5,stride=1,padding=2)
#         # self.conv137_l14=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_1,kernel_size=7,stride=1,padding=3)
#         # self.conv101_l14=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#         # self.conv101Last_l14=nn.Conv2d(in_channels=self.out_ch_line_1*4,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------1 line--4 part modules--------
#             #-----------------------------inception module--------1 line--9 module--------
#         self.conv113_l19=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2,out_channels=self.out_ch_line_1,kernel_size=3,stride=1,padding=1)
#         self.conv512_l19=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2,out_channels=self.out_ch_line_1,kernel_size=5,stride=1,padding=2)
#         self.conv137_l19=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2,out_channels=self.out_ch_line_1,kernel_size=7,stride=1,padding=3)
#         self.conv101_l19=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l19=nn.Conv2d(in_channels=self.out_ch_line_1*4,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------1 line--9 module--------
#             #-----------------------------inception module--------1 line--10 module--------
#         self.conv113_l10=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.num_classes,kernel_size=3,stride=1,padding=1)
#         self.conv512_l10=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.num_classes,kernel_size=5,stride=1,padding=2)
#         self.conv137_l10=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.num_classes,kernel_size=7,stride=1,padding=3)
#         self.conv101_l10=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.num_classes,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l10=nn.Conv2d(in_channels=self.num_classes*4,out_channels=self.num_classes,kernel_size=1,stride=1,padding=0)
#         self.BNline1_last=nn.BatchNorm2d(self.num_classes)
#             #-----------------------------inception module--------1 line--10 module--------
#         #--------------------------------LINE 1 Inception Modules END-----------------------------------
#
#         #--------------------------------LINE 2 Inception Modules-----------------------------------
#             #-----------------------------inception module--------2 line--1 module--------
#         self.conv113_l21=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_2,kernel_size=3,stride=1,padding=1) # int-> 7*3=21 channels, out-> 7*3*7=147 channels
#         self.conv512_l21=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_2,kernel_size=5,stride=1,padding=2)
#         self.conv137_l21=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_2,kernel_size=7,stride=1,padding=3)
#         self.conv101_l21=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l21=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#         self.BN_l2=nn.BatchNorm2d(self.out_ch_line_2)
#             #-----------------------------inception module--------2 line--1 module--------
#             #-----------------------------inception module--------2 line--2 module--------
#         self.conv113_l22=nn.Conv2d(in_channels=self.out_ch_line_2,out_channels=self.out_ch_line_2,kernel_size=3,stride=1,padding=1)
#         self.conv512_l22=nn.Conv2d(in_channels=self.out_ch_line_2,out_channels=self.out_ch_line_2,kernel_size=5,stride=1,padding=2)
#         self.conv137_l22=nn.Conv2d(in_channels=self.out_ch_line_2,out_channels=self.out_ch_line_2,kernel_size=7,stride=1,padding=3)
#         self.conv101_l22=nn.Conv2d(in_channels=self.out_ch_line_2,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l22=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------2 line--2 module--------
#             #-----------------------------inception module--------2 line--3 module--------
#         self.conv113_l23=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=3,stride=1,padding=1)
#         self.conv512_l23=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=5,stride=1,padding=2)
#         self.conv137_l23=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=7,stride=1,padding=3)
#         self.conv101_l23=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l23=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#         # self.conv113_l23=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_2,kernel_size=3,stride=1,padding=1)
#         # self.conv512_l23=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_2,kernel_size=5,stride=1,padding=2)
#         # self.conv137_l23=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_2,kernel_size=7,stride=1,padding=3)
#         # self.conv101_l23=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#         # self.conv101Last_l23=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------2 line--3 module--------
#             #-----------------------------inception module--------2 line--8 module--------
#         self.conv113_l28=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_2,out_channels=self.out_ch_line_2,kernel_size=3,stride=1,padding=1)
#         self.conv512_l28=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_2,out_channels=self.out_ch_line_2,kernel_size=5,stride=1,padding=2)
#         self.conv137_l28=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_2,out_channels=self.out_ch_line_2,kernel_size=7,stride=1,padding=3)
#         self.conv101_l28=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_2,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l28=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------2 line--8 module--------
#         #--------------------------------LINE 2 Inception Modules END-----------------------------------
#
#         #--------------------------------LINE 3 Inception Modules-----------------------------------
#             #-----------------------------inception module--------3 line--1 module--------
#         self.conv113_l31=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_3,kernel_size=3,stride=1,padding=1)
#         self.conv512_l31=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_3,kernel_size=5,stride=1,padding=2)
#         self.conv137_l31=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_3,kernel_size=7,stride=1,padding=3)
#         self.conv101_l31=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l31=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         self.BN_l3=nn.BatchNorm2d(self.out_ch_line_3)
#             #-----------------------------inception module--------3 line--1 module--------
#             #-----------------------------inception module--------3 line--2 module--------
#         self.conv113_l32=nn.Conv2d(in_channels=self.out_ch_line_3,out_channels=self.out_ch_line_3,kernel_size=3,stride=1,padding=1)
#         self.conv512_l32=nn.Conv2d(in_channels=self.out_ch_line_3,out_channels=self.out_ch_line_3,kernel_size=5,stride=1,padding=2)
#         self.conv137_l32=nn.Conv2d(in_channels=self.out_ch_line_3,out_channels=self.out_ch_line_3,kernel_size=7,stride=1,padding=3)
#         self.conv101_l32=nn.Conv2d(in_channels=self.out_ch_line_3,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l32=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------3 line--2 module--------
#             #-----------------------------inception module--------3 line--3 module--------
#         self.conv113_l33=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=3,stride=1,padding=1)
#         self.conv512_l33=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=5,stride=1,padding=2)
#         self.conv137_l33=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=7,stride=1,padding=3)
#         self.conv101_l33=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l33=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         # self.conv113_l34=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=3,stride=1,padding=1)
#         # self.conv512_l34=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=5,stride=1,padding=2)
#         # self.conv137_l34=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=7,stride=1,padding=3)
#         # self.conv101_l34=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         # self.conv101Last_l34=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------3 line--3 module--------
#             #-----------------------------inception module--------3 line--8 module--------
#         self.conv113_l38=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=3,stride=1,padding=1)
#         self.conv512_l38=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=5,stride=1,padding=2)
#         self.conv137_l38=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=7,stride=1,padding=3)
#         self.conv101_l38=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l38=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         # self.conv113_l38=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=3,stride=1,padding=1)
#         # self.conv512_l38=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=5,stride=1,padding=2)
#         # self.conv137_l38=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=7,stride=1,padding=3)
#         # self.conv101_l38=nn.Conv2d(in_channels=self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         # self.conv101Last_l38=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------3 line--8 module--------
#         #--------------------------------LINE 3 Inception Modules END-----------------------------------
#
#         #--------------------------------LINE 4 Inception Modules-----------------------------------
#             #-----------------------------inception module--------4 line--1 module--------
#         self.conv113_l41=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_4,kernel_size=3,stride=1,padding=1)
#         self.conv512_l41=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_4,kernel_size=5,stride=1,padding=2)
#         self.conv137_l41=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_4,kernel_size=7,stride=1,padding=3)
#         self.conv101_l41=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_4,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l41=nn.Conv2d(in_channels=self.out_ch_line_4*4,out_channels=self.out_ch_line_4,kernel_size=1,stride=1,padding=0)
#         self.BN_l4=nn.BatchNorm2d(self.out_ch_line_4)
#             #-----------------------------inception module--------4 line--1 module--------
#             #-----------------------------inception module--------4 line--2 module--------
#         self.conv113_l42=nn.Conv2d(in_channels=self.out_ch_line_4,out_channels=self.out_ch_line_4,kernel_size=3,stride=1,padding=1)
#         self.conv512_l42=nn.Conv2d(in_channels=self.out_ch_line_4,out_channels=self.out_ch_line_4,kernel_size=5,stride=1,padding=2)
#         self.conv137_l42=nn.Conv2d(in_channels=self.out_ch_line_4,out_channels=self.out_ch_line_4,kernel_size=7,stride=1,padding=3)
#         self.conv101_l42=nn.Conv2d(in_channels=self.out_ch_line_4,out_channels=self.out_ch_line_4,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l42=nn.Conv2d(in_channels=self.out_ch_line_4*4,out_channels=self.out_ch_line_4,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------4 line--2 module--------
#             #-----------------------------inception module--------4 line--3 module--------
#         self.conv113_l43=nn.Conv2d(in_channels=self.out_ch_line_4*4,out_channels=self.out_ch_line_4,kernel_size=3,stride=1,padding=1)
#         self.conv512_l43=nn.Conv2d(in_channels=self.out_ch_line_4*4,out_channels=self.out_ch_line_4,kernel_size=5,stride=1,padding=2)
#         self.conv137_l43=nn.Conv2d(in_channels=self.out_ch_line_4*4,out_channels=self.out_ch_line_4,kernel_size=7,stride=1,padding=3)
#         self.conv101_l43=nn.Conv2d(in_channels=self.out_ch_line_4*4,out_channels=self.out_ch_line_4,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_l43=nn.Conv2d(in_channels=self.out_ch_line_4*4,out_channels=self.out_ch_line_4,kernel_size=1,stride=1,padding=0)
#         # self.conv113_l43=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_4,kernel_size=3,stride=1,padding=1)
#         # self.conv512_l43=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_4,kernel_size=5,stride=1,padding=2)
#         # self.conv137_l43=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_4,kernel_size=7,stride=1,padding=3)
#         # self.conv101_l43=nn.Conv2d(in_channels=self.out_ch_line_1+self.out_ch_line_2+self.out_ch_line_3+self.out_ch_line_4,out_channels=self.out_ch_line_4,kernel_size=1,stride=1,padding=0)
#         # self.conv101Last_l43=nn.Conv2d(in_channels=self.out_ch_line_4*4,out_channels=self.out_ch_line_4,kernel_size=1,stride=1,padding=0)
#             #-----------------------------inception module--------4 line--3 module--------
#         #--------------------------------LINE 4 Inception Modules END-----------------------------------
#
#         #--------------transposed convolutions--------------up sampling--------------------------
#         self.tconv26to83=nn.ConvTranspose2d(in_channels=self.out_ch_line_4,out_channels=self.out_ch_line_3,stride=3,padding=0,dilation=3,kernel_size=3,output_padding=1)
#         self.tconv26to126=nn.ConvTranspose2d(in_channels=self.out_ch_line_4,out_channels=self.out_ch_line_2,stride=4,padding=2,dilation=4,kernel_size=8,output_padding=1)
#         self.tconv26to256=nn.ConvTranspose2d(in_channels=self.out_ch_line_4,out_channels=self.out_ch_line_1,stride=6,padding=0,dilation=8,kernel_size=14,output_padding=1)
#
#         self.tconv83to126=nn.ConvTranspose2d(in_channels=self.out_ch_line_3,out_channels=self.out_ch_line_2,stride=1,padding=1,dilation=5,kernel_size=10,output_padding=0)
#         self.tconv83to256=nn.ConvTranspose2d(in_channels=self.out_ch_line_3,out_channels=self.out_ch_line_1,stride=2,padding=0,dilation=6,kernel_size=16,output_padding=1)
#
#         self.tconv126to256=nn.ConvTranspose2d(in_channels=self.out_ch_line_2,out_channels=self.out_ch_line_1,stride=2,padding=1,dilation=2,kernel_size=4,output_padding=1)
#         #--------------transposed convolutions--------------up sampling END----------------------
#         #--------------transposed convolutions--------------up sampling last layer, retain number of features---------------------
#         self.tconv26to83_last=nn.ConvTranspose2d(in_channels=self.out_ch_line_4,out_channels=self.out_ch_line_4,stride=3,padding=0,dilation=3,kernel_size=3,output_padding=1)
#         self.tconv83to126_last=nn.ConvTranspose2d(in_channels=self.out_ch_line_3,out_channels=self.out_ch_line_3,stride=1,padding=1,dilation=5,kernel_size=10,output_padding=0)
#         self.tconv126to256_last=nn.ConvTranspose2d(in_channels=self.out_ch_line_2,out_channels=self.out_ch_line_2,stride=2,padding=1,dilation=2,kernel_size=4,output_padding=1)
#
#
#         #--------------transposed convolutions--------------up sampling last layer, retain number of features END----------------------
#
#
#         #--------------convolutions--------------DOWN sampling------------------------------1 line
#         self.conv256to126_l1=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,stride=2,padding=1,kernel_size=8)
#         self.conv256to83_l1=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,stride=3,padding=1,kernel_size=12)
#         self.conv256to26_l1=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,stride=10,padding=13,kernel_size=32)
#
#         self.conv126to83_l1=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,stride=1,padding=1,kernel_size=46)
#         self.conv126to26_l1=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,stride=5,padding=7,kernel_size=15)
#
#         self.conv83to26_l1=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_1,stride=3,padding=8,kernel_size=24)
#
#         #--------------convolutions--------------DOWN sampling END--------------------------1 line END
#
#         #--------------convolutions--------------DOWN sampling------------------------------mid line
#         self.conv256to126_l2=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_2,stride=2,padding=1,kernel_size=8)
#         self.conv256to83_l2=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_3,stride=3,padding=1,kernel_size=12)
#         self.conv256to26_l2=nn.Conv2d(in_channels=self.out_ch_line_1,out_channels=self.out_ch_line_4,stride=10,padding=13,kernel_size=32)
#
#         self.conv126to83_l2=nn.Conv2d(in_channels=self.out_ch_line_2,out_channels=self.out_ch_line_3,stride=1,padding=1,kernel_size=46)
#         self.conv126to26_l2=nn.Conv2d(in_channels=self.out_ch_line_2,out_channels=self.out_ch_line_4,stride=5,padding=7,kernel_size=15)
#
#         self.conv83to26_l2=nn.Conv2d(in_channels=self.out_ch_line_3,out_channels=self.out_ch_line_4,stride=3,padding=8,kernel_size=24)
#         #--------------convolutions--------------DOWN sampling END--------------------------mid line END
#
#             #-----------------------------LAST inception module 38-------
#         self.conv311_last_38=nn.Conv2d(in_channels=self.out_ch_line_4+self.out_ch_line_3,out_channels=self.out_ch_line_3,kernel_size=3,stride=1,padding=1)
#         self.conv512_last_38=nn.Conv2d(in_channels=self.out_ch_line_4+self.out_ch_line_3,out_channels=self.out_ch_line_3,kernel_size=5,stride=1,padding=2)
#         self.conv137_last_38=nn.Conv2d(in_channels=self.out_ch_line_4+self.out_ch_line_3,out_channels=self.out_ch_line_3,kernel_size=7,stride=1,padding=3)
#         self.conv101_last_38=nn.Conv2d(in_channels=self.out_ch_line_4+self.out_ch_line_3,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_last_38=nn.Conv2d(in_channels=self.out_ch_line_3*4,out_channels=self.out_ch_line_3,kernel_size=1,stride=1,padding=0)
#             #-----------------------------LAST inception module 38--------
#             #-----------------------------LAST inception module 28-------
#         self.conv311_last_28=nn.Conv2d(in_channels=self.out_ch_line_2+self.out_ch_line_3,out_channels=self.out_ch_line_2,kernel_size=3,stride=1,padding=1)
#         self.conv512_last_28=nn.Conv2d(in_channels=self.out_ch_line_2+self.out_ch_line_3,out_channels=self.out_ch_line_2,kernel_size=5,stride=1,padding=2)
#         self.conv137_last_28=nn.Conv2d(in_channels=self.out_ch_line_2+self.out_ch_line_3,out_channels=self.out_ch_line_2,kernel_size=7,stride=1,padding=3)
#         self.conv101_last_28=nn.Conv2d(in_channels=self.out_ch_line_2+self.out_ch_line_3,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_last_28=nn.Conv2d(in_channels=self.out_ch_line_2*4,out_channels=self.out_ch_line_2,kernel_size=1,stride=1,padding=0)
#             #-----------------------------LAST inception module 28--------
#             #-----------------------------LAST inception module 18-------
#         self.conv311_last_28=nn.Conv2d(in_channels=self.out_ch_line_2+self.out_ch_line_1,out_channels=self.out_ch_line_1,kernel_size=3,stride=1,padding=1)
#         self.conv512_last_28=nn.Conv2d(in_channels=self.out_ch_line_2+self.out_ch_line_1,out_channels=self.out_ch_line_1,kernel_size=5,stride=1,padding=2)
#         self.conv137_last_28=nn.Conv2d(in_channels=self.out_ch_line_2+self.out_ch_line_1,out_channels=self.out_ch_line_1,kernel_size=7,stride=1,padding=3)
#         self.conv101_last_28=nn.Conv2d(in_channels=self.out_ch_line_2+self.out_ch_line_1,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#         self.conv101Last_last_28=nn.Conv2d(in_channels=self.out_ch_line_1*4,out_channels=self.out_ch_line_1,kernel_size=1,stride=1,padding=0)
#             #-----------------------------LAST inception module 18--------
#
#
#
#
#
#     def inception_module11(self,input):
#         part1=self.activ(self.conv113_l11(input))
#         part2=self.activ(self.conv512_l11(input))
#         part3=self.activ(self.conv137_l11(input))
#         part4=self.activ(self.conv101_l11(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l1(self.activ(self.conv101Last_l11(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module12(self,input):
#         part1=self.activ(self.conv113_l12(input))
#         part2=self.activ(self.conv512_l12(input))
#         part3=self.activ(self.conv137_l12(input))
#         part4=self.activ(self.conv101_l12(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l1(self.activ(self.conv101Last_l12(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module14(self,input):
#         part1=self.activ(self.conv113_l14(input)) # there
#         # there is a problem RuntimeError: Given groups=1,
#         # weight of size 64 960 3 3, expected input[4, 256, 256, 256] to have 960 channels, but got 256 channels instead
#         part2=self.activ(self.conv512_l14(input))
#         part3=self.activ(self.conv137_l14(input))
#         part4=self.activ(self.conv101_l14(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l1(self.activ(self.conv101Last_l14(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module19(self,input):
#         part1=self.activ(self.conv113_l19(input))
#         part2=self.activ(self.conv512_l19(input))
#         part3=self.activ(self.conv137_l19(input))
#         part4=self.activ(self.conv101_l19(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l1(self.activ(self.conv101Last_l19(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module10(self,input):
#         part1=self.activ(self.conv113_l10(input))
#         part2=self.activ(self.conv512_l10(input))
#         part3=self.activ(self.conv137_l10(input))
#         part4=self.activ(self.conv101_l10(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1) #
#         out=self.BNline1_last(self.activ(self.conv101Last_l10(out)))
#         # out=self.BN_l1(self.activ(self.conv101Last_l10(out))) # batch norm for num of classes # there the error raised
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     #----------------------------LINE 2 BELOW--------------------------
#     def inception_module21(self,input):
#         part1=self.activ(self.conv113_l21(input))
#         part2=self.activ(self.conv512_l21(input))
#         part3=self.activ(self.conv137_l21(input))
#         part4=self.activ(self.conv101_l21(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l2(self.activ(self.conv101Last_l21(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module22(self,input):
#         part1=self.activ(self.conv113_l22(input))
#         part2=self.activ(self.conv512_l22(input))
#         part3=self.activ(self.conv137_l22(input))
#         part4=self.activ(self.conv101_l22(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l2(self.activ(self.conv101Last_l22(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module23(self,input):
#         part1=self.activ(self.conv113_l23(input))
#         part2=self.activ(self.conv512_l23(input))
#         part3=self.activ(self.conv137_l23(input))
#         part4=self.activ(self.conv101_l23(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l2(self.activ(self.conv101Last_l23(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module28(self,input):
#         part1=self.activ(self.conv113_l28(input))
#         part2=self.activ(self.conv512_l28(input))
#         part3=self.activ(self.conv137_l28(input))
#         part4=self.activ(self.conv101_l28(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l2(self.activ(self.conv101Last_l28(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     #----------------------------LINE 3 BELOW--------------------------
#     def inception_module31(self,input):
#         part1=self.activ(self.conv113_l31(input))
#         part2=self.activ(self.conv512_l31(input))
#         part3=self.activ(self.conv137_l31(input))
#         part4=self.activ(self.conv101_l31(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l3(self.activ(self.conv101Last_l31(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module32(self,input):
#         part1=self.activ(self.conv113_l32(input))
#         part2=self.activ(self.conv512_l32(input))
#         part3=self.activ(self.conv137_l32(input))
#         part4=self.activ(self.conv101_l32(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l3(self.activ(self.conv101Last_l32(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module33(self,input):
#         part1=self.activ(self.conv113_l33(input))
#         part2=self.activ(self.conv512_l33(input))
#         part3=self.activ(self.conv137_l33(input))
#         part4=self.activ(self.conv101_l33(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l3(self.activ(self.conv101Last_l33(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module38(self,input):
#         part1=self.activ(self.conv113_l38(input)) # problem is here last 2020-03-11-19:59
#         # RuntimeError: Given groups=1,
#         # weight of size 256 768 3 3, expected input[4, 512, 83, 83] to have 768 channels, but got 512 channels instead
#         part2=self.activ(self.conv512_l38(input))
#         part3=self.activ(self.conv137_l38(input))
#         part4=self.activ(self.conv101_l38(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l3(self.activ(self.conv101Last_l38(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     #----------------------------LINE 4 BELOW--------------------------
#     def inception_module41(self,input):
#         part1=self.activ(self.conv113_l41(input))
#         part2=self.activ(self.conv512_l41(input))
#         part3=self.activ(self.conv137_l41(input))
#         part4=self.activ(self.conv101_l41(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l4(self.activ(self.conv101Last_l41(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module42(self,input):
#         part1=self.activ(self.conv113_l42(input))
#         part2=self.activ(self.conv512_l42(input))
#         part3=self.activ(self.conv137_l42(input))
#         part4=self.activ(self.conv101_l42(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l4(self.activ(self.conv101Last_l42(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#     def inception_module43(self,input):
#         part1=self.activ(self.conv113_l43(input))
#         part2=self.activ(self.conv512_l43(input))
#         part3=self.activ(self.conv137_l43(input))
#         part4=self.activ(self.conv101_l43(input))
#         out=torch.cat((part1,part2,part3,part4),axis=1)
#         out=self.BN_l4(self.activ(self.conv101Last_l43(out)))
#         # out=self.BN(torch.cat((part1,part2,part3,part4),axis=2))
#         return out
#
#
#     def forward(self,input):
#         out=self.inception_module11(input)
#         line1=self.inception_module12(out)
#         line1=self.inception_module12(line1)
#         line2=self.conv256to126_l1(out)
#         line2=self.inception_module21(line2)
#         line2=self.inception_module22(line2)
#         line3=self.conv256to83_l1(out)
#         line3=self.inception_module31(line3)
#         line3=self.inception_module32(line3)
#         line4=self.conv256to26_l1(out)
#         line4=self.inception_module41(line4)
#         line4=self.inception_module42(line4)
#         # print("----------   Debug 1----------")
#         # print("line1.size() ",line1.size())
#         # print("line2.size() ",line2.size())
#         # print("line2.size() ",line3.size())
#         # print("line3.size() ",line4.size())
#         # print("------------------------------")
#         #---------------------------------------------1
#         line1cat=torch.cat([line1,self.tconv26to256(line4),self.tconv83to256(line3),self.tconv126to256(line2)],dim=1)
#         line2cat=torch.cat([self.conv256to126_l2(line1),line2,self.tconv83to126(line3),self.tconv26to126(line4)],dim=1)
#         line3cat=torch.cat([self.conv256to83_l2(line1),self.conv126to83_l2(line2),line3,self.tconv26to83(line4)],dim=1)
#         line4cat=torch.cat([self.conv256to26_l2(line1),self.conv126to26_l2(line2),self.conv83to26_l2(line3),line4],dim=1)
#         # print("----------   Debug 2----------")
#         # print("line1cat.size() ",line1cat.size())
#         # print("line2cat.size() ",line2cat.size())
#         # print("line3cat.size() ",line3cat.size())
#         # print("line4cat.size() ",line4cat.size())
#         # print("------------------------------")
#         # print("debug1")
#         line1=self.inception_module14(line1cat)
#         # print("debug2")
#         line2=self.inception_module23(line2cat)
#         # print("debug3")
#         line3=self.inception_module33(line3cat)
#         # print("debug4")
#         line4=self.inception_module43(line4cat)
#         # print("debug5")
#         #---------------------------------------------1
#         line1=self.inception_module12(line1)
#         line1=self.inception_module12(line1)
#         line2=self.inception_module22(line2)
#         line2=self.inception_module22(line2)
#         line3=self.inception_module32(line3)
#         line3=self.inception_module32(line3)
#         line4=self.inception_module42(line4)
#         line4=self.inception_module42(line4)
#         #---------------------------------------------2
#         line1cat=torch.cat([line1,self.tconv126to256(line2),self.tconv83to256(line3),self.tconv26to256(line4)],dim=1)
#         line2cat=torch.cat([self.conv256to126_l2(line1),line2,self.tconv83to126(line3),self.tconv26to126(line4)],dim=1)
#         line3cat=torch.cat([self.conv256to83_l2(line1),self.conv126to83_l2(line2),line3,self.tconv26to83(line4)],dim=1)
#         line4cat=torch.cat([self.conv256to26_l2(line1),self.conv126to26_l2(line2),self.conv83to26_l2(line3),line4],dim=1)
#         line1=self.inception_module14(line1cat)
#         line2=self.inception_module23(line2cat)
#         line3=self.inception_module33(line3cat)
#         line4=self.inception_module43(line4cat)
#         #---------------------------------------------2
#         line1=self.inception_module12(line1)
#         line2=self.inception_module22(line2)
#         line3=self.inception_module32(line3)
#         line4=self.inception_module42(line4)
#         #---------------------------------upsampling tail below
#         line3cat=torch.cat([line3,self.tconv26to83_last(line4)],dim=1)
#         line3=self.inception_module38(line3cat)
#         line2cat=torch.cat([line2,self.tconv83to126_last(line3)],dim=1)
#         line2=self.inception_module28(line2cat)
#         line1cat=torch.cat([line1,self.tconv126to256_last(line2)],dim=1)
#         line1=self.inception_module19(line1cat)
#         out=self.inception_module10(line1) # problem is here
#         # running_mean should contain 7 elements not 64
#
#         # line1cat=torch.cat([line1,line2,line3,line4],dim=1)
#         # line1cat=torch.cat([line1,line2,line3,line4],dim=1)
#         # line1cat=torch.cat([line1,line2,line3,line4],dim=1)
#         #---------------------------------------------------------------------------
#         # out=inceptionModule(self.first_layer_in_channels*self.ch_per_soruce,self.layer_out_channels).forward_inception(input)
#         # out=self.conv2d11(input)
#         # out=self.midBatchNorm(out)
#         # out=self.activ(out)
#         # side=out
#         # print("side size=",side.size())
#         # out=self.conv2d12(input) # <<<<<---------------- problem RuntimeError: Given groups=1, weight of size 64 1 9 9, expected input[4, 64, 256, 256] to have 1 channels, but got 64 channels instead
#         # out=self.midBatchNorm(out)
#         # out=self.activ(out)
#         # print("pre conv size=",out.size())
#         # out=self.deconv11(out)
#         # print("after conv size=",out.size())
#         # out=self.activ(out)
#         # out=self.midBatchNorm(out)
#         # out=torch.cat([out,side],dim=1)  #<<<<<--------Sizes of tensors must match except in dimension 1. Got 250 and 256 in dimension 2
#         # out=self.conv2dend(out)
#         # out=self.outBatchNorm(out)
#         # out=self.activ(out)
#         return out
#     def load_weights(self, weights_file):
#         other, ext = os.path.splitext(weights_file)
#         if ext == '.pkl' or '.pth':
#             print('Loading weights into state dict...')
#             self.load_state_dict(torch.load(weights_file, map_location=lambda storage, loc: storage))
#             # stirct false\true -
#             # stirct false\true - allows us to load parameters from certain place
#
#             print('Finished!')
#         else:
#             print('Sorry only .pth and .pkl files supported.')
#
# #-------HYPER PARAMS-------------
# batch_size=2
# num_workers=2
# momentum=0.9
# weight_decay=1e-9
# # loss=nn.CrossEntropyLoss()
# loss=nn.MSELoss()
#
# learn_rate=3e-3
#
# gpu_ids=[6,7]
# device_ids = device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
# #-------HYPER PARAMS-------------
#
# data_set=Dataset_custom() # maybe i should send it to cuda?
# data_train=data.DataLoader(data_set,batch_size=batch_size,num_workers=num_workers)
# # Gnet=Gmodel()
# # Dnet=Dmodel()
# # Goptimizer=torch.optim.SGD(Gnet.parameters(),lr=learn_rate,weight_decay=weight_decay)
# # Doptimizer=torch.optim.SGD(Dnet.parameters(),lr=learn_rate,weight_decay=weight_decay)
# train=True
# HRnet1=HRnet(7,3)
# # HRnet1=HRnet1.cuda()
# # HRnet1=HRnet1.cuda() # worked!
# # device_ids = [6, 7]
# # device = torch.device(device_ids)
# HRnet1=HRnet1.to(device)
# HRnet1 = nn.DataParallel(HRnet1, gpu_ids)
#
#
# optimizer=torch.optim.SGD(HRnet1.parameters(),lr=learn_rate,weight_decay=weight_decay)
# loss=nn.CrossEntropyLoss()
#
# # print("GPU issues")
# # use_cuda = torch.cuda.is_available()
#
# pixels_classes = np.array(
#     [[255, 255, 255, 255],
#      [250, 10, 222, 255],
#      [0, 255, 0, 255],
#      [255, 0, 0, 255],
#      [50, 0, 0, 255],
#      [30, 0, 0, 255],
#      [0, 0, 255, 255]])
#
# epochs=400
# step_save_params=20 # per epochs
# step_save_check_out=1 # per epochs for batch
# print(torch.cuda.is_available())
# #-----------test save in uncreated direcotry----------uncerated folder---------------
# # save_folder = "/storage/3050/MazurM/db/2Parameters/Params1/HRnetCustomV1/Parameters/"
# # folder_name="uncreated folder"
# # numberer=234
# # new_path=save_folder+"/"+folder_name+str(numberer)+"/"
# # try:
# #     os.mkdir(new_path)
# # except FileExistsError:
# #     pass
# # # save_folder = "/home/std_11/MazurM/UnetV1/Parameters/"
# # # save_folder="/storage/MazurM/Task1/SavedParameters/UnetV1/polygon/"
# # # file_saved_parameters = "epoch" + str(1) + "err_" + str(2) + "_Unet1v1Params.pth"
# # file_saved_parameters = "epoch" + str(1) + "err_" + str(2) + "_Unet1v1Params.pth"
# # torch.save(HRnet1.state_dict(), os.path.join(new_path, file_saved_parameters))
# # print("check saved parameters")
# #---------test save in uncreated direcotry----------uncerated folder---------------
#
#
# for ep in range(1,epochs):
#     calc_start_time = datetime.datetime.now()
#     error_epoch=0.0
#     samples_epoch=0
#     saved=False
#     print("CURRENT EPOCH ", ep)
#
#     for i, (input,mask) in enumerate(data_train):
#         # print("workin, current sample:",i)
#         # input=input.to(device_ids)
#         # mask=mask.to(device_ids)
#         # print("outside dataset class") #
#         # print("outside after gpu") #
#         input=input.to(device)
#         mask=mask.to(device) # torch.Size([4, 256, 256])
#         optimizer.zero_grad()
#         # print("input.is_cuda=",input.is_cuda) # false  #
#         # print("mask.is_cuda=",mask.is_cuda) # false  #
#         net_output=HRnet1(input) # <<<<<<<<<<<<<<< torch.Size([4, 7, 256, 256])
#         # print("net_output.size()",net_output.size())  #
#         # print("mask.size()",mask.size())  #
#         error=loss(net_output,mask) # RuntimeError: 1only batches of spatial targets supported (non-empty 3D tensors) but got targets of size: : [4, 256, 256, 1]
#         error.backward()
#         # error_print=error.item()[0]
#         optimizer.step()
#         error_epoch += error.data.cpu().numpy().item()
#         samples_epoch+=batch_size
#         # print("debug mark 1 end")  #
#         _, predicted = torch.max(net_output.data, 1)
#         #-------painted output-----------
#         # save_folder_img = "/storage/3050/MazurM/db/2Parameters/Params1/HRnetCustomV1/Parameters/masks/"
#         # size_p=predicted.size()
#         # for i in range(size_p[0]):
#         #     mask_img=np.zeros((size_p[1],size_p[2],3),dtype=np.uint8)
#         #     predicted_mask=np.zeros((size_p[1],size_p[2],3),dtype=np.uint8)
#         #     for w in range(size_p[1]):
#         #         for h in range(size_p[2]):
#         #             index_of_pix_in_dict=predicted[i,w,h].item()
#         #             index_of_pix_in_dict_3ch=predicted[i,w,h].item()
#         #             channels=pixels_classes[predicted[i,w,h].item()][0:3]
#         #             predicted_mask[w,h]=pixels_classes[predicted[i,w,h].item()][0:3]
#         #             #----------------
#         #             mask_img[w,h]=pixels_classes[mask[i,w,h].item()][0:3]
#         #
#         #     plt.imshow(mask_img)
#         #     plt.show()
#         #     mplot.image.imsave(save_folder_img+str(i)+"Predict.png",predicted_mask)
#         #     mplot.image.imsave(save_folder_img+str(i)+"GroundTruth.png",mask_img)
#
#         #-------painted output-----------
#         # error_epoch=((error_epoch*1.0)/(samples_epoch*1.0)) # delete, just for check arithmetics
#         # print("debug mark 2 end")  #
#
#         if (not (ep%step_save_check_out) and not(saved)):
#             saved=True
#             save_folder_img = "/storage/3050/MazurM/db/2Parameters/Params1/HRnetCustomV1/Parameters/masks/"
#             size_p = predicted.size()
#             for i_sample in range(size_p[0]):
#                 mask_img = np.zeros((size_p[1], size_p[2], 3), dtype=np.uint8)
#                 predicted_mask = np.zeros((size_p[1], size_p[2], 3), dtype=np.uint8)
#                 for w in range(size_p[1]):
#                     for h in range(size_p[2]):
#                         index_of_pix_in_dict = predicted[i_sample, w, h].item()
#                         index_of_pix_in_dict_3ch = predicted[i_sample, w, h].item()
#                         channels = pixels_classes[predicted[i_sample, w, h].item()][0:3]
#                         predicted_mask[w, h] = pixels_classes[predicted[i_sample, w, h].item()][0:3]
#                         # ----------------
#                         mask_img[w, h] = pixels_classes[mask[i_sample, w, h].item()][0:3]
#             mplot.image.imsave(save_folder_img+"epoch_"+str(ep)+"_num_"+str(i)+"Predict.png",predicted_mask)
#             mplot.image.imsave(save_folder_img+"epoch_"+str(ep)+"_num_"+str(i)+"GroundTruth.png",mask_img)
#
#     error_epoch=((error_epoch*1.0)/(samples_epoch*1.0))
#     # print("epochs ",ep,"; error=",error_epoch)
#     print(" error ",error_epoch)
#     if(not(ep%step_save_params)):
#         save_folder = "/storage/3050/MazurM/db/2Parameters/Params1/HRnetCustomV1/Parameters/"
#         # save_folder = "/home/std_11/MazurM/UnetV1/Parameters/"
#         # save_folder="/storage/MazurM/Task1/SavedParameters/UnetV1/polygon/"
#         file_saved_parameters = "ep" + str(ep) + "_error_" + str(error_epoch) + "_HRnet1v1.pth"
#         torch.save(HRnet1.state_dict(), os.path.join(save_folder, file_saved_parameters))
#         print("check saved parameters")
#
#     calc_end_time = datetime.datetime.now()
#     difference=calc_end_time-calc_start_time
#     seconds_in_day = 24 * 60 * 60
#     datetime.timedelta(0, 8, 562000)
#     print_time=divmod(difference.days * seconds_in_day + difference.seconds, 60)
#     print("time was spent:",print_time[0]," minutes,",print_time[1]," seconds")
#
#     # if(not(ep%step_save_check_out)):
#     #     save_folder_img = "/storage/3050/MazurM/db/2Parameters/Params1/HRnetCustomV1/Parameters/masks/"
#     #     size_p=predicted.size()
#     #     for i in range(size_p[0]):
#     #         mask_img=np.zeros((size_p[1],size_p[2],3),dtype=np.uint8)
#     #         predicted_mask=np.zeros((size_p[1],size_p[2],3),dtype=np.uint8)
#     #         for w in range(size_p[1]):
#     #             for h in range(size_p[2]):
#     #                 index_of_pix_in_dict=predicted[i,w,h].item()
#     #                 index_of_pix_in_dict_3ch=predicted[i,w,h].item()
#     #                 channels=pixels_classes[predicted[i,w,h].item()][0:3]
#     #                 predicted_mask[w,h]=pixels_classes[predicted[i,w,h].item()][0:3]
#     #                 #----------------
#     #                 mask_img[w,h]=pixels_classes[mask[i,w,h].item()][0:3]
#
#             # plt.imshow(mask_img)
#             # plt.show()
#             # mplot.image.imsave(save_folder_img+"epoch_"+str(ep)+"_num_"+str(i)+"Predict.png",predicted_mask)
#              # mplot.image.imsave(save_folder_img+"epoch_"+str(ep)+"_num_"+str(i)+"GroundTruth.png",mask_img)
#
#         # pass
#
# print("Check before you go")
# print("Have you checked outputs?")
# print("ok")
#
# # if (train):
# #
# #     GAN=GANmodel() # workable 1
# #     Generator=GAN.Generator.parameters() # workable 1
# #     Discriminator=GAN.Discriminator.parameters() # workable 1
# #
# #     # Discriminator=GANmodel().Discriminator # workable 2
# #     # Generator=GANmodel().Generator # workable 2
# #
# #     Goptimizer=torch.optim.SGD(Generator,lr=learn_rate,weight_decay=weight_decay)
# #     Doptimizer=torch.optim.SGD(Discriminator,lr=learn_rate,weight_decay=weight_decay)
# #     #--------------------------------------------
#


#
# # GAN.train
# for ep in range(epochs):
#     for i, (noise,realA) in enumerate(data_train):
#         #target=realA - both are synonyms
#         fakeA,DoutFake,DoutReal=GAN(noise,realA)
#         Doptimizer.zero_grad()
#         labelsFake = torch.zeros(DoutFake.size())
#         labelsFake
#
#         GeneratorError=torch.log(labelsFake-DoutFake)
#
#
#         pass
#
#         # #-------------------------------------------------------
#         # #-------------------------------------------------------
#         # print("debug") # noise requires_grad=False
#         # Dnet.train()
#         # Gnet.train()
#         # Doptimizer.zero_grad()
#         # fakeA=Gnet(noise)
#         # DoutFake=Dnet(fakeA)
#         # labelsFake = torch.zeros(fakeA.size())
#         # GeneratorError=torch.log(labelsFake-DoutFake)
#         # #--------------
#         # # target - realA
#         # DoutReal=Dnet(target)
#         # DtotalError=torch.log(DoutFake)+torch.log(torch.ones(DoutReal.size())-DoutReal)
#         # DtotalError.backward(retain_graph=True)
#         # Doptimizer.step()
#         # Goptimizer.zero_grad()
#         # GeneratorError.backward()
#         # Goptimizer.step()
#         # # Dtotal
#         # #-------------------------------------------------------
#         # #-------------------------------------------------------
# #
# #         print(fakeA.size()) # torch.Size([4, 3, 256, 256])
#
#
#         # labelsFake=
#         # GeneratorError=
#
#
#         # pass
#
#
#         # #-----training the discriminator---------
#         # Doptimizer.zero_grad()
#         # Gout=Gnet(noise).size()
#         # #--------FAKE
#         # DoutFake=Dnet(Gnet(noise).detach())
#         # print("GFake.size()=",Gout) # torch.Size([4, 3, 256, 256])
#         # print("noise.size()=",noise.size()) #torch.Size([4, 1, 256, 256])
#         # print("target.size()=",target.size()) #torch.Size([4, 256, 256, 4])
#         # lables_fake=torch.zeros(DoutFake.size()) #
#         # DlossFake=loss(DoutFake,lables_fake)
#         # # DlossFake.backward()
#         # #--------REAL
#         # DoutReal=Dnet(target)
#         # labels_real=torch.ones(DoutReal.size())
#         # DlossReal=loss(DoutReal,labels_real)
#         # #-----total error for Discriminator
#         # Dtotal=(DlossReal+DlossReal)*0.5-0.5 # HOW TO EVALUATE LOG and 1-LOG here?
#         # # WE CANT MINIMIZE ABOVE, because it must aimed to 0.5 but not to ZERO! so we should use here
#         # # can we use -0.5 ????
#         # # Dtotal=
#         #
#         #
#         #
#         # Goptimizer.zero_grad()
#         # Goptimizer.step()
#         #
#         #
#         #
#         # noise.requires_grad=True
#         # #-----training the discriminator---------
#         # Goptimizer.zero_grad()
#         # # noise=noise.cuda()
#         # # target=target.cuda()
#         # # Dmodel.eval()
#         # print(Dout.size())
#         # Dloss=loss(Dout,lables_fake)
#         # Dloss.backward()
#         # Doptimizer.step()
#         # Goptimizer.step()
#         # #--------next half of training---
#         # Dout=Dnet(target)
#         # DlossReal=loss(Dout,1)
#         # DlossReal.backward()
#         # Doptimizer.step()
#         # print("done, Enter smth")
#         # x=input()
#
#         #---------below is just for sake of backup !!!!!!!!!!!!!!
#         #-----training the discriminator---------
#
#         #-----training the discriminator---------
#         # # Doptimizer.zero_grad()
#         # # Goptimizer.zero_grad()
#         # # # noise=noise.cuda()
#         # # # target=target.cuda()
#         # # noise.requires_grad=True
#         # # # Dmodel.eval()
#         # # Dout=Dnet(Gnet(noise))
#         # # print(Dout.size())
#         # # lables_fake=torch.ones(Dout.size())
#         # # Dloss=loss(Dout,lables_fake)
#         # # Dloss.backward()
#         # # Doptimizer.step()
#         # # Goptimizer.step()
#         # # #--------next half of training---
#         # # Dout=Dnet(target)
#         # # DlossReal=loss(Dout,1)
#         # # DlossReal.backward()
#         # # Doptimizer.step()
#         # # print("done, Enter smth")
#         # # x=input()
#
#
#

print("DEBUG")



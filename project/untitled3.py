# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:48:29 2021

@author: Neng Xiong
"""
import os
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt

rgb_path = os.path.join('train', 'images', 'rgb')
nir_path = os.path.join('train', 'images', 'nir')
bou_path = os.path.join('train', 'boundaries')
mask_path = os.path.join('train', 'masks')
l1_path = os.path.join('train', 'labels', 'cloud_shadow')
l2_path = os.path.join('train', 'labels', 'double_plant')
l3_path = os.path.join('train', 'labels', 'planter_skip')
l4_path = os.path.join('train', 'labels', 'standing_water')
l5_path = os.path.join('train', 'labels', 'waterway')
l6_path = os.path.join('train', 'labels', 'weed_cluster')
files_rgb = os.listdir(rgb_path) 
files_nir = os.listdir(nir_path) 
files_bou = os.listdir(bou_path) 
files_mask = os.listdir(mask_path) 
files_l1 = os.listdir(l1_path) 
files_l2 = os.listdir(l2_path) 
files_l3 = os.listdir(l3_path) 
files_l4 = os.listdir(l4_path) 
files_l5 = os.listdir(l5_path) 
files_l6 = os.listdir(l6_path) 
#files_rgb = os.listdir(rgb_path) 
#%%
color_matrix = np.zeros([len(files_rgb),4])
label_array = np.zeros(len(files_rgb))

for i in range(len(files_rgb)):
    print (i)
    img_rgb = mpimg.imread(os.path.join(rgb_path, files_rgb[i]))
    img_nir = mpimg.imread(os.path.join(nir_path, files_nir[i]))
    img_bou = mpimg.imread(os.path.join(bou_path, files_bou[i]))
    img_mask = mpimg.imread(os.path.join(mask_path, files_mask[i]))
    img_l1 = mpimg.imread(os.path.join(l1_path, files_l1[i]))
    img_l2 = mpimg.imread(os.path.join(l2_path, files_l2[i]))
    img_l3 = mpimg.imread(os.path.join(l3_path, files_l3[i]))
    img_l4 = mpimg.imread(os.path.join(l4_path, files_l4[i]))
    img_l5 = mpimg.imread(os.path.join(l5_path, files_l5[i]))
    img_l6 = mpimg.imread(os.path.join(l6_path, files_l6[i]))
    final_img = np.zeros([img_rgb.shape[0],img_rgb.shape[1],img_rgb.shape[2]+1]);
    final_img[:,:,0] = img_rgb[:,:,0] * img_bou * img_mask
    final_img[:,:,1] = img_rgb[:,:,1] * img_bou * img_mask
    final_img[:,:,2] = img_rgb[:,:,2] * img_bou * img_mask
    final_img[:,:,3] = img_nir * img_bou * img_mask
    c1 = (1 in img_l1); c2 = (1 in img_l2); c3 = (1 in img_l3);
    c4 = (1 in img_l4); c5 = (1 in img_l5); c6 = (1 in img_l6);
    color_matrix[i,0] = np.mean(final_img[np.nonzero(final_img[:,:,0])[0],np.nonzero(final_img[:,:,0])[1],0])
    color_matrix[i,1] = np.mean(final_img[np.nonzero(final_img[:,:,1])[0],np.nonzero(final_img[:,:,1])[1],1])
    color_matrix[i,2] = np.mean(final_img[np.nonzero(final_img[:,:,2])[0],np.nonzero(final_img[:,:,2])[1],2])
    color_matrix[i,3] = np.mean(final_img[np.nonzero(final_img[:,:,3])[0],np.nonzero(final_img[:,:,3])[1],3])
    if(c1==True):
        label_array[i] = 1;
    elif(c2==True):
        label_array[i] = 2;
    elif(c3==True):
        label_array[i] = 3;
    elif(c4==True):
        label_array[i] = 4;
    elif(c5==True):
        label_array[i] = 5;
    elif(c6==True):
        label_array[i] = 6;
        
     
        
     
#%%
for i in range(3):
    print (i)
    img_rgb = mpimg.imread(os.path.join(rgb_path, files_rgb[i]))
    img_nir = mpimg.imread(os.path.join(nir_path, files_nir[i]))
    img_bou = mpimg.imread(os.path.join(bou_path, files_bou[i]))
    img_mask = mpimg.imread(os.path.join(mask_path, files_mask[i]))
    img_l1 = mpimg.imread(os.path.join(l1_path, files_l1[i]))
    img_l2 = mpimg.imread(os.path.join(l2_path, files_l2[i]))
    img_l3 = mpimg.imread(os.path.join(l3_path, files_l3[i]))
    img_l4 = mpimg.imread(os.path.join(l4_path, files_l4[i]))
    img_l5 = mpimg.imread(os.path.join(l5_path, files_l5[i]))
    img_l6 = mpimg.imread(os.path.join(l6_path, files_l6[i]))
    final_img = np.zeros([img_rgb.shape[0],img_rgb.shape[1],img_rgb.shape[2]+1]);
    
    final_img[:,:,0] = img_rgb[:,:,0] * img_bou * img_mask
    final_img[:,:,1] = img_rgb[:,:,1] * img_bou * img_mask
    final_img[:,:,2] = img_rgb[:,:,2] * img_bou * img_mask
    final_img[:,:,3] = img_nir * img_bou * img_mask
    
    # c1 = (1 in img_l1); c2 = (1 in img_l2); c3 = (1 in img_l3);
    # c4 = (1 in img_l4); c5 = (1 in img_l5); c6 = (1 in img_l6);
    # color_matrix[i,0] = np.mean(final_img[np.nonzero(final_img[:,:,0]),0])
    # color_matrix[i,1] = np.mean(final_img[np.nonzero(final_img[:,:,1]),1])
    # color_matrix[i,2] = np.mean(final_img[np.nonzero(final_img[:,:,2]),2])
    # color_matrix[i,3] = np.mean(final_img[np.nonzero(final_img[:,:,3]),3])
    # if(c1==True):
    #     label_array[i] = 1;
    # elif(c2==True):
    #     label_array[i] = 2;
    # elif(c3==True):
    #     label_array[i] = 3;
    # elif(c4==True):
    #     label_array[i] = 4;
    # elif(c5==True):
    #     label_array[i] = 5;
    # elif(c6==True):
    #     label_array[i] = 6;
#%%
colors = ["orange","steelblue","darkgreen","red","skyblue","maroon"]
lnames = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster']
fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in range(6):
    sub_matrix = color_matrix[label_array==i+1,:]
    ax1.scatter(sub_matrix[:,1],sub_matrix[:,2],c=colors[i],s=0.5,label=lnames[i])
ax1.legend(markerscale=10)
ax1.set_ylabel("Blue")
ax1.set_xlabel("Green")    
fig.savefig('./figs/green_blue.png',dpi=1200)  
plt.show();

#%%
# final_img = np.zeros([img_rgb.shape[0],img_rgb.shape[1],img_rgb.shape[2]+1]);
# final_img[:,:,0] = img_rgb[:,:,0] * img_bou * img_mask
# final_img[:,:,1] = img_rgb[:,:,1] * img_bou * img_mask
# final_img[:,:,2] = img_rgb[:,:,2] * img_bou * img_mask
# final_img[:,:,3] = img_nir * img_bou * img_mask
# r = np.mean(final_img[np.nonzero(final_img[:,:,0]),0])
# g = np.mean(final_img[np.nonzero(final_img[:,:,1]),1])
# b = np.mean(final_img[np.nonzero(final_img[:,:,2]),2])
# infe = np.mean(final_img[np.nonzero(final_img[:,:,3]),3])
# #%%
# labels 
# c1 = (1 in img_l1);
# c2 = (1 in img_l2);
# c3 = (1 in img_l3);
# c4 = (1 in img_l4);
# c5 = (1 in img_l5);
# c6 = (1 in img_l6);
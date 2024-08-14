import numpy as np
import matplotlib.pyplot as plt


#%%
unshrinking = True
ccdnum = 13
if unshrinking:
    pickle_file = 'defringed-ccd-%02d.npy' % ccdnum
else:
    pickle_file = 'defringed-nounshrinking-ccd-%02d.npy' % ccdnum

def load_data_from_npy(filename):
    data = {}
    with open(filename, 'rb') as f:
        data['g_images'] = np.load(f)       # input ori image (npix, nobj)
        data['fimages'] = np.load(f)        # raw defringe (remove background & median fringe pattern) (npix, nobj)
        data['g_elximages'] = np.load(f)    # exilir defringe image (traditional)
        data['Z'] = np.load(f)              # fringes pattern result (ours)
        data['median'] = np.load(f)         # median
        data['masks'] = np.load(f)          # masks
        data['pca_defringe'] = np.load(f)   # load my defringe image
    return data

# 调用函数读取数据
data = load_data_from_npy(pickle_file)

# 输出数据维度
# print('g_ori_image:', np.shape(data['g_ori_image']))
print('g_images:', np.shape(data['g_images']))
print('fimages:', np.shape(data['fimages']))
print('g_elximages:', np.shape(data['g_elximages']))
print('Z:', np.shape(data['Z']))
print('masks:', np.shape(data['masks']))
print('pca_defringe:', np.shape(data['pca_defringe']))

## 提取第一列
width = 500
height = 500
show_column = 2
# g_ori_image = data['g_ori_image'][:,show_column].reshape(width, height)
g_image = data['g_images'][:,show_column].reshape(width, height)
fimage = data['fimages'][:,show_column].reshape(width, height)
g_elximage = data['g_elximages'][:,show_column].reshape(width, height)
z = data['Z'][:,show_column].reshape(width, height)
mask = data['masks'][:,show_column].reshape(width, height)
pca_defringe = data['pca_defringe'][:,show_column].reshape(width, height)

## 将原图展示出来

# 创建一个图像画布
fig, axes = plt.subplots(1, 6, figsize=(15, 6))
# 显示每一列的图像
axes[0].imshow(g_image, cmap='gray')
axes[0].set_title('g.images')
axes[0].axis('off')

axes[1].imshow(fimage, cmap='gray')
axes[1].set_title('fimages(raw defringe)(remove bg & md)')
axes[1].axis('off')

axes[2].imshow(g_elximage, cmap='gray')
axes[2].set_title('elx defringe')
axes[2].axis('off')

axes[3].imshow(z, cmap='gray')
axes[3].set_title('fringe pattern(ours)')
axes[3].axis('off')

axes[4].imshow(mask, cmap='gray')
axes[4].set_title('mask')
axes[4].axis('off')

axes[5].imshow(pca_defringe, cmap='gray')
axes[5].set_title('pca_defringe')
axes[5].axis('off')

# axes[5].imshow(g_ori_image, cmap='gray')
# axes[5].set_title('g_ori_image')
# axes[5].axis('off')

plt.tight_layout()
plt.show()


#%%
pickle_file_2 = 'defringed-ccd-13_rpca.npy'

def load_data_from_npy(filename):
    data = {}
    with open(filename, 'rb') as f:
        data['L'] = np.load(f)    # ori (the very ori)
        data['S'] = np.load(f)    # input ori image (npix, nobj)
    return data

# 调用函数读取数据
data = load_data_from_npy(pickle_file_2)

# 输出数据维度
print('L:', np.shape(data['L']))
print('S:', np.shape(data['S']))

## 提取第一列
width = 500
height = 500
show_column = 1
decomp_L = data['L'][:,show_column].reshape(width, height)
decomp_S = data['S'][:,show_column].reshape(width, height)

# 创建一个图像画布
fig, axes = plt.subplots(1, 2)
# 显示每一列的图像
axes[0].imshow(decomp_L, cmap='gray')
axes[0].set_title('decomp_L')
axes[0].axis('off')

axes[1].imshow(decomp_S, cmap='gray')
axes[1].set_title('decomp_S')
axes[1].axis('off')

plt.tight_layout()
plt.show()
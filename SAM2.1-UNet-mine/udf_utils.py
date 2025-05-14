# Author: wxk
# Time: 2022/1/17 16:00
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import shutil


def visualization(img, save_path=''):
    for i in range(img.shape[1]):
        show_img = img[0, i, :, :]
        print(show_img.shape)
        show_img = show_img.cpu()
        array1 = show_img.detach().numpy()
        # maxValue = array1.max()
        # array1 = array1 * 255 / maxValue
        mat = np.uint8(array1)  # 将浮点数转换为 8 位整数（0~255）
        # mat = mat.transpose(1, 2, 0)
        # mat = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img', mat)
        cv2.imwrite(save_path.replace('.jpg', '_' + str(i) + '.jpg'), mat)
        cv2.waitKey(0)


def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # 删除目录及其内容
        os.mkdir(path)
    else:
        os.mkdir(path)


def mask_overlap(image_path, mask_path, anno_path, save_path):
    mkr(save_path)  # 先创建保存目录

    # 掩码颜色映射：当像素值在 [128, 255] 时，使用 mask_colors[2]（红色）。
    # 标注颜色映射：当像素值在 [128, 255] 时，使用 anno_colors[2]（深天蓝色）。
    mask_colors = ['None', 'yellow', 'red']  # 定义掩码的颜色
    anno_colors = ['white', 'yellow', 'deepskyblue']  # 人工标注的颜色
    bounds = [0, 128, 256]  # 设定颜色映射的界限

    # mpl.colors.ListedColormap 定义颜色映射方案
    # mpl.colors.BoundaryNorm 定义数值与颜色的对应关系
    mask_cmap = mpl.colors.ListedColormap(mask_colors)
    anno_cmap = mpl.colors.ListedColormap(anno_colors)
    norm = mpl.colors.BoundaryNorm(bounds, mask_cmap.N)

    img_list = os.listdir(image_path)
    img_list.sort()
    for img in img_list:
        print('drawing', img, '...')
        image = Image.open(os.path.join(image_path, img))
        mask = Image.open(os.path.join(mask_path, img.replace('jpg', 'png')))
        anno = Image.open(os.path.join(anno_path, img.replace('jpg', 'png')))
        anno = np.asarray(anno)[:, :, 0]  # 将多通道图像提取为单通道图像，防止后续出现问题

        plt.figure(figsize=(7.2, 4.8))
        plt.axis('off')  # 关闭坐标轴
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)  # 去掉边框

        plt.imshow(image, aspect='auto')
        plt.imshow(anno, aspect='auto', alpha=0.5, interpolation='none', cmap=anno_cmap, norm=norm)
        plt.imshow(mask, aspect='auto', alpha=0.5, interpolation='none', cmap=mask_cmap, norm=norm)
        plt.savefig(os.path.join(save_path, img.replace('jpg', 'png')))  # 保存结果
        plt.close()


import os
from sklearn.metrics import mean_absolute_error, precision_recall_curve, precision_recall_fscore_support
import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy


# 计算掩码 (mask_file) 和标注 (anno_file) 之间的误差
def evalutate(mask_file, anno_file):
    mask = cv2.imread(mask_file)[:, :, 0]  # cv2 读的是 BGR ，0 代表的是 B
    mask = mask.astype(int)  # 转换为整数类型
    anno = cv2.imread(anno_file)[:, :, 0]  # B
    anno = anno.astype(int)
    size = mask.shape
    # 将 2D 图像展平为 1D 数组
    mask = mask.reshape(size[0] * size[1], )
    anno = anno.reshape(size[0] * size[1], )
    # threshold：设置阈值缩放到 0 和 1，precise：计算像素差 mae/255
    mae_mode = 'threshold'
    # mae_mode = 'precise'

    precision, recall, thresholds = precision_recall_curve(anno, mask, pos_label=255)
    Fscore = []
    for i in range(len(precision)):
        if precision[i] == 0 or recall[i] == 0:
            continue
        Fscore.append(2 / ((1 / precision[i]) + (1 / recall[i])))  # 计算公式
    Fmax = max(Fscore)
    Favg = np.mean(Fscore)
    print('Fmax:', Fmax)
    print('Favg:', Favg)

    MAE = mean_absolute_error(anno, mask) / 255
    print('MAE:', MAE)

    return Fmax, Favg, MAE


# 批量评估整个数据集的掩码质量，并计算平均 F-score 和 MAE。
def evalutate_saliency(mask_path, anno_path):
    anno_files = os.listdir(anno_path)
    anno_files.sort()
    Fmax_list = []
    Favg_list = []
    MAE_list = []
    count = 0

    for file in anno_files:
        print('\nevaluating', file)
        mask_file = os.path.join(mask_path, file)
        anno_file = os.path.join(anno_path, file)
        Fmax, Favg, MAE = evalutate(mask_file, anno_file)
        Fmax_list.append(Fmax)
        Favg_list.append(Favg)
        MAE_list.append(MAE)
        count += 1
    print(count, 'files evaluated')
    print('\nTOTAL Fmax:', np.mean(Fmax_list), 'TOTAL Favg:', np.mean(Favg_list), 'TOTAL MAE:', np.mean(MAE_list))


# 该函数将多个缺陷类型的显著性图（dent、scratch、crack 等）合并成一张最终显著性图。
def sum_saliency_map():
    path_all = ''
    path_save = ''
    path_dent = ''
    path_scratch = ''
    path_crack = ''
    path_glass_shatter = ''
    path_lamp_broken = ''
    path_tire_flat = ''
    mkr(path_save)
    list_path = [path_dent, path_scratch, path_crack, path_glass_shatter, path_lamp_broken, path_tire_flat]
    for file in sorted(os.listdir(path_all)):
        print('summing', file)
        img = cv2.imread(os.path.join(path_all, file))[:, :, 0]
        (height, width) = img.shape
        slice = np.zeros((height, width))
        for path in list_path:
            if file in os.listdir(path):
                pred = cv2.imread(os.path.join(path, file))[:, :, 0]
                slice = np.maximum(slice, pred)  # 比较两个 array，逐个元素取最大
        sum_img = np.zeros(shape=(height, width, 3), dtype=np.float32)
        sum_img[:, :, 0] = slice[:, :]
        sum_img[:, :, 1] = slice[:, :]
        sum_img[:, :, 2] = slice[:, :]
        sum_img = sum_img.astype(np.uint8)
        plt.imsave(os.path.join(path_save, file), sum_img)


if __name__ == '__main__':
    image_path = '/data1_hdd/gyy/CarDD/instance_image/test/1'
    mask_path = '/data1_hdd/gyy/CarDD/results/SAM2-UNet/instance/1/test_mask'
    anno_path = '/data1_hdd/gyy/CarDD/instance_mask/test/1'
    save_path = '/data1_hdd/gyy/CarDD/results/SAM2-UNet/instance/1/test_overlap'
    mask_overlap(image_path, mask_path, anno_path, save_path)
    evalutate_saliency(mask_path, anno_path)  # anno 是 GT

    # sum_saliency_map()

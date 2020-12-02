import os
import numpy as np
from keras.preprocessing import image
from skimage.transform import resize
from skimage.measure import label, regionprops
from PIL import Image
from DiscCrop_UNet import DeepModel
from util import BW_img, disc_crop


DiscSeg_size = 640
DiscCrop_size = 512


img_dir = 'H:/Glaucoma/REFUGE/original/REFUGE2-Test/'
#mask_dir = 'H:/AMD/Annotation-DF-Training400/Training400/Disc_Masks_unified/'
disc_save_dir = 'H:/Glaucoma/REFUGE/crop512/test/img/'
#mask_save_dir = 'H:/AMD/process/crop448/train/mask/'
coord_save_dir = "H:/Glaucoma/REFUGE/crop512/test/coord/"

if not os.path.exists(disc_save_dir):
    os.makedirs(disc_save_dir)
img_list = [file for file in os.listdir(img_dir) if file.lower().endswith('.jpg')]
print(len(img_list))

DiscCrop_model = DeepModel(size_set=DiscSeg_size)
DiscCrop_model.load_weights('DiscCrop_UNet.h5')


for i in range(len(img_list)):
    temp_txt = img_list[i]
    print(temp_txt)

    # Step 1: Crop disc and save coord
    org_img = np.asarray(image.load_img(img_dir + temp_txt))
    #org_mask = np.asarray(image.load_img(mask_dir + temp_txt))[:, :, 0]

    # Disc region detection by U-Net
    temp_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3)) * 255
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    disc_map = DiscCrop_model.predict([temp_img])
    disc_map = BW_img(np.reshape(disc_map, (DiscSeg_size, DiscSeg_size)), 0.5)

    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)

    disc_region, err_coord, crop_coord = disc_crop(org_img, DiscCrop_size, C_x, C_y)
    #mask_region, _, _ = disc_crop(org_mask, DiscCrop_size, C_x, C_y, img_mode="L")

    disc_result = Image.fromarray(disc_region)
    disc_result.save(disc_save_dir + temp_txt[:-4] + '.jpg')
    #mask_result = Image.fromarray(mask_region)
    #mask_result.save(mask_save_dir + temp_txt[:-4] + '.jpg')
    coord = [err_coord[0], err_coord[1], err_coord[2], err_coord[3], crop_coord[0], crop_coord[1], crop_coord[2],
             crop_coord[3]]
    print(coord)
    np.savetxt(coord_save_dir + temp_txt[:-4] + '.txt', coord, fmt="%d")

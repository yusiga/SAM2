CUDA_VISIBLE_DEVICES="6" \
python test.py \
--checkpoint "/data1_hdd/gyy/CarDD/cp/adapter/v1/SAM2-UNet-50.pth" \
--test_image_path "/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-TE/CarDD-TE-Image/" \
--test_gt_path "/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-TE/CarDD-TE-Mask/" \
--save_path "/data1_hdd/gyy/CarDD/results/adapter/v1/test_mask/"
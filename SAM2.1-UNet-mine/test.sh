CUDA_VISIBLE_DEVICES="6" \
python test.py \
--checkpoint "/data1_hdd/gyy/CarDD/cp/SAM2-UNet/instance/1/SAM2-UNet-50.pth" \
--test_image_path "/data1_hdd/gyy/CarDD/instance_image/test/1/" \
--test_gt_path "/data1_hdd/gyy/CarDD/instance_mask/test/1/" \
--save_path "/data1_hdd/gyy/CarDD/results/SAM2-UNet/instance/1/test_mask/"
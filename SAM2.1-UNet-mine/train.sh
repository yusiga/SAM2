CUDA_VISIBLE_DEVICES="6" \
python train.py \
--hiera_path "/data1_ssd/gyy/CarDD/code/SOD/SAM2.1-UNet/model/sam2.1_hiera_large.pt" \
--train_image_path "/data1_hdd/gyy/CarDD/instance_image/train/1/" \
--train_mask_path "/data1_hdd/gyy/CarDD/instance_mask/train/1/" \
--save_path "/data1_hdd/gyy/CarDD/cp/SAM2-UNet/instance/1/" \
--epoch 50 \
--lr 0.001 \
--batch_size 12
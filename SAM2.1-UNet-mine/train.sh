CUDA_VISIBLE_DEVICES="6" \
python train.py \
--hiera_path "/data1_ssd/gyy/CarDD/code/adapter/v1/SAM2.1-UNet-mine/model/sam2.1_hiera_large.pt" \
--train_image_path "/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-TR/CarDD-TR-Image/" \
--train_mask_path "/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-TR/CarDD-TR-Mask/" \
--save_path "/data1_hdd/gyy/CarDD/cp/adapter/v1/" \
--epoch 50 \
--lr 0.001 \
--batch_size 12
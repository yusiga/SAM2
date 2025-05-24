CUDA_VISIBLE_DEVICES="6" \
python train.py \
--hiera_path "/data1_ssd/gyy/CarDD/code/adapter/SAM2.1-UNet-mine/model/sam2.1_hiera_large.pt" \
--train_image_path "/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-TR/CarDD-TR-Image/" \
--train_mask_path "/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-TR/CarDD-TR-Mask/" \
--val_image_path "/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-VAL/CarDD-VAL-Image/" \
--val_mask_path "/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-VAL/CarDD-VAL-Mask/" \
--save_path "/data1_hdd/gyy/CarDD/cp/adapter/v1/" \
--epoch 50 \
--lr 0.0001 \
--batch_size 8
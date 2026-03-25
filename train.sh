CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "checkpoints/sam2.1_hiera_tiny.pt" \
--save_path "output" \
--epoch 100 \
--lr 0.001 \
--batch_size 4

#--train_image_path "<set your training image dir here>" \
#--train_mask_path "<set your training mask dir here>" \
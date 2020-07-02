CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net_hospital.py --dataset hospital \
--cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_hospital.yaml --bs 8 --nw 8 \
--output_dir /home/lianjie/mask-rcnn.pytorch-1.0/jeline_output/eccv/dcn_fpn/gk_yy_ours
# --load_ckpt /home/lianjie/mask-rcnn.pytorch-1.0/jeline_output/competition/ckpt/model_step7499.pth \
# --resume


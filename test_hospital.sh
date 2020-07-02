
 CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/test_net_hospital.py  --dataset hospital --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_hospital.yaml \
 --load_ckpt /home/lianjie/mask-rcnn.pytorch-1.0/jeline_output/eccv/dcn_fpn/gk_yy_ours/ckpt/model_step19499.pth \
 --output_dir /home/lianjie/mask-rcnn.pytorch-1.0/jeline_output/test  --multi-gpu-testing  --overwrite  # --multi-gpu-testing    # --overwrite

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/test_net_hospital.py  --dataset hospital --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_hospital.yaml \
# --load_ckpt /home/lianjie/mask-rcnn.pytorch-1.0/jeline_output/product_base/ckpt/model_step79999.pth \
# --output_dir /home/lianjie/mask-rcnn.pytorch-1.0/jeline_output/test  --multi-gpu-testing  --overwrite

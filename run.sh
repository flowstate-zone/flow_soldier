# conda init
# conda activate clustering

# Swin Base
# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train.py --config_file configs/flowstate/urbnsurf-melbourne.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH './log/swin_small.pth'  OUTPUT_DIR './log/flowstate/urbnsurf-melbourne'
# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train.py --config_file configs/flowstate/surftown-munich.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH './log/swin_small.pth'  OUTPUT_DIR './log/flowstate/surftown-munich'
# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train.py --config_file configs/flowstate/all.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH './log/swin_small.pth'  OUTPUT_DIR './log/flowstate/all'
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train.py --config_file configs/flowstate/reviewed_sessions.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH './log/swin_small.pth'  OUTPUT_DIR './log/flowstate/reviewed_sessions'


# Swin Small# './log/swin_small_tea.pth'
#CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/msmt17/swin_small.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH 'path/to/SOLIDER/log/lup/swin_small/checkpoint_tea.pth' OUTPUT_DIR './log/msmt17/swin_small' SOLVER.BASE_LR 0.0002 SOLVER.OPTIMIZER_NAME 'SGD' MODEL.SEMANTIC_WEIGHT 0.2

# Swin Tiny
#CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/msmt17/swin_tiny.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH 'path/to/SOLIDER/log/lup/swin_tiny/checkpoint_tea.pth' OUTPUT_DIR './log/msmt17/swin_tiny' SOLVER.BASE_LR 0.0008 SOLVER.OPTIMIZER_NAME 'SGD' MODEL.SEMANTIC_WEIGHT 0.2
#MODEL.RESUME './log/flowstate/urbnsurf-melbourne/transformer_60.pth'
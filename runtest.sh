# CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/flowstate/all.yml TEST.WEIGHT './log/flowstate/urbnsurf-melbourne/transformer_60.pth' MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/flowstate/all.yml TEST.WEIGHT './log/swin_small_tea.pth' MODEL.SEMANTIC_WEIGHT 0.2

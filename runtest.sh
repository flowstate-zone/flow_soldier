CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/flowstate/swin_small.yml TEST.WEIGHT './log/swin_small_market.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2

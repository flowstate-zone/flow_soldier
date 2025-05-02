from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import torch.distributed as dist
import boto3
import json
import zlib


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_reviewed_dataset(cfg):
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(cfg.S3_BUCKET)

    # create the dataset root directory
    if not os.path.exists(cfg.DATASETS.ROOT_DIR):
        os.makedirs(cfg.DATASETS.ROOT_DIR)

    # for object_summary in my_bucket.objects.filter():
    reviewed_sessions = []
    for object_summary in my_bucket.objects.filter(Prefix=cfg.S3_REVIEWED_DATA):
        if object_summary.key.endswith('.json'):
            # download the json file from s3
            my_bucket.download_file(object_summary.key, os.path.join(cfg.DATASETS.ROOT_DIR,object_summary.key.split('/')[-1]))
            # open the json file
            with open(os.path.join(cfg.DATASETS.ROOT_DIR,object_summary.key.split('/')[-1]), 'r') as f:
                # load the json data
                data = json.load(f)
                data['zip'] = object_summary.key.replace('.json', '.tar.gz')
                reviewed_sessions.append(data)
    # reviewed_sessions = [reviewed_sessions[0], reviewed_sessions[-1]]

    # create train test split here
    # if zlib.adler32(group.session.values[0].encode()) % 9 == 0:
    test_sessions = [session for session in reviewed_sessions if zlib.adler32(session['zip'].encode()) % 4 == 0]
    # print(len(test_sessions))
    train_sessions = [session for session in reviewed_sessions if zlib.adler32(session['zip'].encode()) % 4 != 0]
    # print(len(train_sessions))
            
    # find sessions in the listed datasets
    dataset_list = set(cfg.DATASETS.TRAIN_NAMES + cfg.DATASETS.VAL_NAMES)

    for dataset in dataset_list:
        pool_slug = dataset.split('/')[0]
        session_type = dataset.split('/')[1]
        # download all the tar.gz files from s3
        for session in test_sessions:
            if session['pool_slug'] == pool_slug and session['session_type'] == session_type:
                session_name = session['zip'].split('/')[-1].replace('.tar.gz', '')
                print(f"session_name: {session_name}")
                # create the session directory
                if not os.path.exists(os.path.join(cfg.DATASETS.ROOT_DIR.replace('train', 'test'), pool_slug,session_type, session_name)):
                    os.makedirs(os.path.join(cfg.DATASETS.ROOT_DIR.replace('train', 'test'), pool_slug,session_type, session_name))
                # download the zip
                my_bucket.download_file(session['zip'], os.path.join(cfg.DATASETS.ROOT_DIR.replace('train', 'test'), session['zip'].split('/')[-1]))
                # extract the zip
                os.system('tar -xzf {} -C {}'.format(os.path.join(cfg.DATASETS.ROOT_DIR.replace('train', 'test'), session['zip'].split('/')[-1]), os.path.join(cfg.DATASETS.ROOT_DIR.replace('train', 'test'), pool_slug,session_type, session_name)))
        
        for session in train_sessions:
            if session['pool_slug'] == pool_slug and session['session_type'] == session_type:
                session_name = session['zip'].split('/')[-1].replace('.tar.gz', '')
                print(f"session_name: {session_name}")
                # create the session directory
                if not os.path.exists(os.path.join(cfg.DATASETS.ROOT_DIR, pool_slug,session_type, session_name)):
                    os.makedirs(os.path.join(cfg.DATASETS.ROOT_DIR, pool_slug,session_type, session_name))
                # download the zip
                my_bucket.download_file(session['zip'], os.path.join(cfg.DATASETS.ROOT_DIR, session['zip'].split('/')[-1]))
                # extract the zip
                os.system('tar -xzf {} -C {}'.format(os.path.join(cfg.DATASETS.ROOT_DIR, session['zip'].split('/')[-1]), os.path.join(cfg.DATASETS.ROOT_DIR, pool_slug,session_type, session_name)))

    # reviewed_sessions = [session for session in reviewed_sessions if '{}/{}'.format(session['pool_slug'], session['session_type']) in dataset_list]
    
    # dataset_names = ['{}/{}'.format(session['pool_slug'], session['session_type']) for session in reviewed_sessions]
    # # get the set dataset names
    # dataset_names = set(dataset_names)
    # print(f"dataset_names: {dataset_names}")

    # # 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    # pull the reviewed data and structure it for training
    get_reviewed_dataset(cfg)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    #  logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    (
        train_loader,
        train_loader_normal,
        val_loaders,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )

    if cfg.MODEL.RESUME != "":
        logger.info(f"Resuming from {cfg.MODEL.RESUME}")
        model.load_param(cfg.MODEL.RESUME)

    # pretrained = '/home/ubuntu/Code/flow_soldier/last.pth'
    # print(pretrained)
    # if pretrained != "":
    #     logger.info(f"Resuming from {pretrained}")
    #     checkpoint = torch.load(pretrained, weights_only=True, map_location='cuda')
    #     model.load_state_dict(checkpoint['model'], strict=False)
    #     if cfg.MODEL.DEVICE_ID != -1:
    #         model = model.cuda()

    print(f"num_classes:{num_classes}")
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    if cfg.SOLVER.WARMUP_METHOD == "cosine":
        logger.info("===========using cosine learning rate=======")
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info("===========using normal learning rate=======")
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            cfg.SOLVER.WARMUP_FACTOR,
            cfg.SOLVER.WARMUP_EPOCHS,
            cfg.SOLVER.WARMUP_METHOD,
        )

    # if pretrained != "":
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     scheduler.step(checkpoint['epoch'])

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loaders,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        args.local_rank,
        # start_epoch=checkpoint['epoch'] if pretrained != "" else 0,
    )

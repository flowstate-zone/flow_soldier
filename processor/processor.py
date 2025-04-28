import logging
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import boto3


def do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loaders,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    local_rank,
    wandb_logger=None,
    bucket=None,
    s3_output_dir=None,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    s3 = boto3.client('s3')

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            vid_local = vid.numpy()
            if len(set(vid_local)) == 1:
                logger.info("Skip batch, only one class")
                continue
            # if len(vid_local) < cfg.SOLVER.IMS_PER_BATCH:
            #     logger.info(
            #         f"Skip batch, batch size {len(vid_local)} less than cfg.SOLVER.IMS_PER_BATCH {cfg.SOLVER.IMS_PER_BATCH}"
            #     )
            #     continue

            # if n_iter > 500:
                # break

            # print unique views
            # unique_views = np.unique(target_view.numpy())
            # print(f"Unique views in batch: {unique_views}")

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)

            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat, _ = model(
                    img, label=target, cam_label=target_cam, view_label=target_view
                )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = (
                            scheduler._get_lr(epoch)[0]
                            if cfg.SOLVER.WARMUP_METHOD == "cosine"
                            else scheduler.get_lr()[0]
                        )
                        logger.info(
                            "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                                epoch,
                                (n_iter + 1),
                                len(train_loader),
                                loss_meter.avg,
                                acc_meter.avg,
                                base_lr,
                            )
                        )
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = (
                        scheduler._get_lr(epoch)[0]
                        if cfg.SOLVER.WARMUP_METHOD == "cosine"
                        else scheduler.get_lr()[0]
                    )
                    logger.info(
                        "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                            epoch,
                            (n_iter + 1),
                            len(train_loader),
                            loss_meter.avg,
                            acc_meter.avg,
                            base_lr,
                        )
                    )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == "cosine":
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info(
                "Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch,
                    time_per_batch * (n_iter + 1),
                    train_loader.batch_size / time_per_batch,
                )
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)
                        ),
                    )
            else:
                # save checkpoint
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)
                    ),
                )
                # cp to s3
                # s3.upload_file(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)), bucket, os.path.join(s3_output_dir, cfg.MODEL.NAME + "_{}.pth".format(epoch)))
                
                # save the latest checkpoint
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR, "last.pth"
                    ),
                )
                # print all files in the output dir
                logger.info("All files in the output dir:")
                for root, dirs, files in os.walk(cfg.OUTPUT_DIR):
                    for file in files:
                        logger.info(os.path.join(root, file))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for val_loader_dict in val_loaders:
                        val_loader = val_loader_dict["loader"]
                        num_query = val_loader_dict["num_query"]
                        name = val_loader_dict["name"]
                        logger.info("")
                        logger.info("")
                        logger.info(f"Validation on {name.upper()} set")
                        evaluator = R1_mAP_eval(
                            num_query,
                            max_rank=50,
                            feat_norm=cfg.TEST.FEAT_NORM,
                        )
                        for n_iter, (
                            img,
                            vid,
                            camid,
                            camids,
                            target_view,
                            _,
                        ) in enumerate(val_loader):
                            with torch.no_grad():
                                img = img.to(device)
                                camids = camids.to(device)
                                target_view = target_view.to(device)
                                feat, _ = model(
                                    img, cam_label=camids, view_label=target_view
                                )
                                evaluator.update((feat, vid, camid))
                        cmc, mAP, _, _, _, _, _ = evaluator.compute()
                        logger.info("Validation Results - Epoch: {}".format(epoch))
                        logger.info("mAP: {:.1%}".format(mAP))
                        for r in [1, 5, 10]:
                            logger.info(
                                "CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1])
                            )
                        torch.cuda.empty_cache()
            else:
                model.eval()
                for val_loader_dict in val_loaders:
                    val_loader = val_loader_dict["loader"]
                    num_query = val_loader_dict["num_query"]
                    name = val_loader_dict["name"]
                    logger.info("")
                    logger.info("")
                    logger.info(f"Validation on {name.upper()} set")
                    evaluator = R1_mAP_eval(
                        num_query,
                        max_rank=50,
                        feat_norm=cfg.TEST.FEAT_NORM,
                    )
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(
                        val_loader
                    ):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, _ = model(
                                img, cam_label=camids, view_label=target_view
                            )
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    if wandb_logger is not None:
                        wandb_logger.log({'val/mAP': mAP})
                    for r in [1, 5, 10]:
                        logger.info(
                            "CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1])
                        )
                        if wandb_logger is not None:
                            wandb_logger.log({f'val/Rank-{r}': cmc[r - 1]})
                    torch.cuda.empty_cache()


def do_inference(cfg, model, val_loaders):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")
    for val_loader_dict in val_loaders:
        val_loader = val_loader_dict["loader"]
        num_query = val_loader_dict["num_query"]
        name = val_loader_dict["name"]
        logger.info("")
        logger.info("")
        logger.info(f"##=> Validation on {name.upper()} <=##")

        evaluator = R1_mAP_eval(
            num_query,
            max_rank=50,
            feat_norm=cfg.TEST.FEAT_NORM,
            reranking=cfg.TEST.RE_RANKING,
        )

        evaluator.reset()

        if device:
            if torch.cuda.device_count() > 1:
                print("Using {} GPUs for inference".format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(device)

        model.eval()
        img_path_list = []

        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(
            val_loader
        ):
            with torch.no_grad():
                img = img.to(device)
                camids = camids.to(device)
                target_view = target_view.to(device)
                feat, _ = model(img, cam_label=camids, view_label=target_view)
                evaluator.update((feat, pid, camid))
                img_path_list.extend(imgpath)

        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

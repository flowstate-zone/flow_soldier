import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import (
    RandomIdentitySampler,
    RandomIdentitySampler_IdUniform,
    RandomSessionSampler,
)
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
from .flowstate_purchased import FlowstatePurchased, FlowstateSessions
import torch.distributed as dist
from itertools import chain
from .mm import MM

__factory = {
    "market1501": Market1501,
    "msmt17": MSMT17,
    "mm": MM,
    "flowstate_purchased": FlowstatePurchased,
    "flowstate_sessions": FlowstateSessions,   
}


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return (
        torch.stack(imgs, dim=0),
        pids,
        camids,
        viewids,
    )


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(
                probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"
            ),
        ]
    )

    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS

    train_datasets = []
    pid_offset = 0
    sid_offset = 0
    # make sure to increment pid and sid so there isn't overlap between datasets
    for dataset_name in sorted(cfg.DATASETS.TRAIN_NAMES):
        # dataset = FlowstatePurchased(
        #     root=cfg.DATASETS.ROOT_DIR,
        #     dataset_name=dataset_name,
        #     include_val=False,
        #     train_limit=cfg.DATASETS.TRAIN_LIMIT,
        #     pid_offset=pid_offset,
        #     sid_offset=sid_offset,
        # )
        dataset = FlowstateSessions(
            root=cfg.DATASETS.ROOT_DIR,
            dataset_name=dataset_name,
            include_val=False,
            train_limit=cfg.DATASETS.TRAIN_LIMIT,
            pid_offset=pid_offset,
            sid_offset=sid_offset,
        )
        pid_offset += dataset.num_train_pids
        sid_offset += dataset.num_train_vids
        train_datasets.append(dataset)

    # concatenate all dataset.train together not using torch
    full_train = list(chain.from_iterable(dataset.train for dataset in train_datasets))
    train_set = ImageDataset(full_train, train_transforms)
    train_set_normal = ImageDataset(full_train, val_transforms)
    num_classes = sum(dataset.num_train_pids for dataset in train_datasets)
    cam_num = sum(dataset.num_train_cams for dataset in train_datasets)
    view_num = sum(dataset.num_train_vids for dataset in train_datasets)

    if cfg.DATALOADER.SAMPLER in ["softmax_triplet", "img_triplet"]:
        print("using img_triplet sampler")
        if cfg.MODEL.DIST_TRAIN:
            print("DIST_TRAIN START")
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(
                full_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
            )
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                data_sampler, mini_batch_size, True
            )
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:

            train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomSessionSampler(
                    full_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
                ),
                num_workers=num_workers,
                collate_fn=train_collate_fn,
                drop_last=True,
            )
            # train_loader = DataLoader(
            #     train_set,
            #     batch_size=cfg.SOLVER.IMS_PER_BATCH,
            #     sampler=RandomIdentitySampler(
            #         full_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
            #     ),
            #     num_workers=num_workers,
            #     collate_fn=train_collate_fn,
            # )
    elif cfg.DATALOADER.SAMPLER == "softmax":
        print("using softmax sampler")
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
        )
    elif cfg.DATALOADER.SAMPLER in ["id_triplet", "id"]:
        print("using ID sampler")
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler_IdUniform(
                full_train, cfg.DATALOADER.NUM_INSTANCE
            ),
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            drop_last=True,
        )
    else:
        print(
            "unsupported sampler! expected softmax or triplet but got {}".format(
                cfg.SAMPLER
            )
        )

    val_loaders = []
    for dataset_name in sorted(cfg.DATASETS.VAL_NAMES):
        # val_dataset = FlowstatePurchased(
        #     root=cfg.DATASETS.ROOT_DIR,
        #     dataset_name=dataset_name,
        #     include_train=False,
        #     val_limit=cfg.DATASETS.VAL_LIMIT,
        # )
        val_dataset = FlowstateSessions(
            root=cfg.DATASETS.ROOT_DIR,
            dataset_name=dataset_name,
            include_train=False,
            val_limit=cfg.DATASETS.VAL_LIMIT,
        )
        val_set = ImageDataset(val_dataset.query + val_dataset.gallery, val_transforms)
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
        )
        val_loaders.append(
            {
                "loader": val_loader,
                "name": dataset_name,
                "num_query": len(val_dataset.query),
            }
        )

    train_loader_normal = DataLoader(
        train_set_normal,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )
    return (
        train_loader,
        train_loader_normal,
        val_loaders,
        num_classes,
        cam_num,
        view_num,
    )

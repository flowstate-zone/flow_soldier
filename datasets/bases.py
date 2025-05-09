from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                    img_path
                )
            )
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        if len(data) > 0:
            min_pid = min(pids)
            max_pid = max(pids)
            min_view = min(tracks)
            max_view = max(tracks)
        else:
            min_pid = max_pid = min_view = max_view = None

        return (
            num_pids,
            num_imgs,
            num_cams,
            num_views,
            min_pid,
            max_pid,
            min_view,
            max_view,
        )

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        (
            num_train_pids,
            num_train_imgs,
            num_train_cams,
            num_train_views,
            min_pid,
            max_pid,
            min_view,
            max_view,
        ) = self.get_imagedata_info(train)
        print("getting query")
        num_query_pids, num_query_imgs, num_query_cams, num_train_views, _, _, _, _ = (
            self.get_imagedata_info(query)
        )
        (
            num_gallery_pids,
            num_gallery_imgs,
            num_gallery_cams,
            num_train_views,
            _,
            _,
            _,
            _,
        ) = self.get_imagedata_info(gallery)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info(
            "  train    | {:5d} | {:8d} | {:9d}".format(
                num_train_pids, num_train_imgs, num_train_cams
            )
        )
        logger.info(
            "  query    | {:5d} | {:8d} | {:9d}".format(
                num_query_pids, num_query_imgs, num_query_cams
            )
        )
        logger.info(
            "  gallery  | {:5d} | {:8d} | {:9d}".format(
                num_gallery_pids, num_gallery_imgs, num_gallery_cams
            )
        )
        logger.info("  ----------------------------------------")
        logger.info(
            f"  train    | min_pid: {min_pid} | max_pid: {max_pid} | min_view: {min_view} | max_view: {max_view}"
        )
        logger.info("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path
        #  return img, pid, camid, trackid,img_path.split('/')[-1]

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import random


class FlowstatePurchased(BaseImageDataset):
    """
    Market1501 structure
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    """

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super(FlowstatePurchased, self).__init__()
        self.dataset_dir = root
        self.cam_path_to_id_map = None
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_train_dir(self.train_dir, relabel=True)
        query, pid_container = self._process_query_gallery_dir(
            self.query_dir, relabel=False
        )
        gallery, _ = self._process_query_gallery_dir(
            self.gallery_dir, relabel=False, pid_container=pid_container
        )

        if verbose:
            print("=> FlowstatePurchased loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
            self.num_train_vids,
        ) = self.get_imagedata_info(self.train)
        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
            self.num_query_vids,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
            self.num_gallery_vids,
        ) = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _get_cam_path(self, img_path):
        cam_path = "_".join(img_path.split("/")[-3:-1])
        return cam_path

    def _cam_path_to_id_map(self, dir_path):
        if self.cam_path_to_id_map is None:
            img_paths = glob.glob(osp.join(dir_path, "*/*/*.jpg"))
            unique_cam_paths = set(self._get_cam_path(x) for x in img_paths)
            print(f"found {len(unique_cam_paths)} unique cam paths")
            cam_path_to_id_map = {
                cam_path: i for i, cam_path in enumerate(unique_cam_paths)
            }
            print(f"cam_path_to_id_map: {cam_path_to_id_map}")
            self.cam_path_to_id_map = cam_path_to_id_map
        return self.cam_path_to_id_map

    def _process_train_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*/*/*.jpg"))

        pid_container = set()
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)
            # get pid
            pid = filename.split("_")[0]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        cam_path_to_id_map = self._cam_path_to_id_map(dir_path)
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)
            # get pid
            pid = filename.split("_")[0]

            # get camid
            cam_path = self._get_cam_path(img_path)
            camid = cam_path_to_id_map[cam_path]

            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid, 1))
        return dataset

    def _process_query_gallery_dir(
        self, dir_path, relabel=False, pid_container=None, max_pids=100
    ):
        def get_img_user_id(img_path):
            return img_path.split("/")[-1].split("_")[0]

        if pid_container is not None:
            kept_pids = pid_container
        else:
            kept_pids = []

        dataset = []

        for cam in self.cam_path_to_id_map.keys():
            print(cam)
            pool, pool_zone = cam.split("_")[0], cam.split("_")[1]
            img_paths = glob.glob(osp.join(dir_path, f"{pool}/{pool_zone}/*.jpg"))

            if pid_container is None:
                # make the sampling reproduceable
                random.seed(1)
                user_ids = list(set(get_img_user_id(x) for x in img_paths))
                # sample max pids
                pid_sample = random.sample(user_ids, min(max_pids, len(user_ids)))
                kept_pids.extend(pid_sample)

            pid2label = {pid: label for label, pid in enumerate(kept_pids)}
            cam_path_to_id_map = self._cam_path_to_id_map(dir_path)
            for img_path in sorted(img_paths):
                pid = get_img_user_id(img_path)
                if pid not in kept_pids:
                    continue
                # get camid
                cam_path = self._get_cam_path(img_path)
                camid = cam_path_to_id_map[cam_path]
                if relabel:
                    pid = pid2label[pid]
                dataset.append((img_path, pid, camid, 1))
        return dataset, kept_pids

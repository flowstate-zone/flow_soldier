import glob

import os.path as osp

from .bases import BaseImageDataset
import random
import logging

logger = logging.getLogger("transreid.train")


class FlowstatePurchased(BaseImageDataset):
    """
    For Flowstate pool data.
    Market1501 structure
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    """

    def __init__(
        self,
        root="",
        verbose=True,
        dataset_name="",
        include_train=True,
        include_val=True,
        train_limit=None,
        val_limit=50,
        **kwargs,
    ):
        super(FlowstatePurchased, self).__init__()
        self.dataset_dir = osp.join(root, dataset_name)
        logger.info(f"Loading {dataset_name} from {self.dataset_dir}")
        self.cam_path_to_id_map = None
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")

        self._check_before_run()

        if include_train:
            pid_container = self._get_pid_container(self.train_dir, limit=train_limit)
            train = self._process_dir(self.train_dir, pid_container, relabel=True)
        else:
            train = []

        if include_val:
            pid_container = self._get_pid_container(self.query_dir, limit=val_limit)
            query = self._process_dir(
                self.query_dir, pid_container, relabel=False, is_query=True
            )
            gallery = self._process_dir(self.gallery_dir, pid_container, relabel=False)
        else:
            query = []
            gallery = []

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print("=> FlowstatePurchased loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

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

    def _get_pid_container(self, dir_path, limit=None):
        glob_path = osp.join(dir_path, "*.jpg")
        img_paths = glob.glob(glob_path)
        pid_container = set()
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)
            # get pid
            pid = filename.split("_")[0]
            pid_container.add(pid)
        if limit is not None:
            random.seed(0)
            pid_container = random.sample(pid_container, min(limit, len(pid_container)))
        logger.info(f"Found {len(pid_container)} ids in {dir_path}")
        return pid_container

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

    def _process_dir(
        self,
        dir_path,
        pid_container,
        relabel=False,
        is_query=False,
    ):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)
            # get pid
            pid = filename.split("_")[0]
            if pid not in pid_container:
                continue

            camid = 0 if is_query else 1
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid, 1))
        return dataset

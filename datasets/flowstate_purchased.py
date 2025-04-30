import glob
import hashlib
import os.path as osp

from .bases import BaseImageDataset
import random
import logging
import math
import pandas as pd
import zlib

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
        pid_offset=0,
        sid_offset=0,
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
            session_container = self._get_session_container(
                self.train_dir, limit=train_limit
            )
            pid_container = self._get_pid_container(self.train_dir, session_container)
            train = self._process_dir(
                self.train_dir,
                pid_container,
                session_container,
                relabel=True,
                pid_offset=pid_offset,
                sid_offset=sid_offset,
            )
        else:
            train = []

        if include_val:
            session_container = self._get_session_container(
                self.query_dir, limit=val_limit
            )
            pid_container = self._get_pid_container(self.query_dir, session_container)
            query = self._process_dir(
                self.query_dir,
                pid_container,
                session_container,
                relabel=False,
                is_query=True,
            )
            gallery = self._process_dir(
                self.gallery_dir, pid_container, session_container, relabel=False
            )
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
            _,
            _,
            _,
            _,
        ) = self.get_imagedata_info(self.train)

        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
            self.num_query_vids,
            _,
            _,
            _,
            _,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
            self.num_gallery_vids,
            _,
            _,
            _,
            _,
        ) = self.get_imagedata_info(self.gallery)

    def _get_pid_container(self, dir_path, session_container, limit=None):
        glob_path = osp.join(dir_path, "*.jpg")
        img_paths = glob.glob(glob_path)
        pid_container = set()
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)
            # get pid
            pid = filename.split("__")[0]
            # session_id = filename.split("__")[0]
            # if session_id not in session_container:
            #     continue
            pid_container.add(pid)
        if limit is not None:
            random.seed(0)
            pid_container = random.sample(pid_container, min(limit, len(pid_container)))
        logger.info(f"Found {len(pid_container)} ids in {dir_path}")
        return pid_container

    def _get_session_container(self, dir_path, limit=None):
        glob_path = osp.join(dir_path, "*.jpg")
        img_paths = glob.glob(glob_path)
        session_container = set()
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)
            # get pid
            session_id = 0
            session_container.add(session_id)
        if limit is not None:
            session_container = sorted(list(session_container))[:limit]
        logger.info(f"Found {len(session_container)} sessions in {dir_path}")
        return session_container

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
        session_container,
        pid_offset=0,
        sid_offset=0,
        relabel=False,
        is_query=False,
    ):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pid2label = {pid: label + pid_offset for label, pid in enumerate(pid_container)}
        session2label = {
            session: label + sid_offset
            for label, session in enumerate(session_container)
        }
        dataset = []
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)
            # get pid
            pid = filename.split("__")[0]
            session_id = 0
            if session_id not in session_container:
                continue
            
            wave_time = filename.split("__")[1]
            camid = int(hashlib.md5(wave_time.encode()).hexdigest(), 16) % 256
            if relabel:
                pid = pid2label[pid]
                sid = session2label[session_id]
            else:
                sid = 1
            dataset.append((img_path, pid, camid, sid))
        return dataset

class FlowstateSessions(BaseImageDataset):
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
        pid_offset=0,
        sid_offset=0,
        **kwargs,
    ):
        super(FlowstateSessions, self).__init__()
        self.dataset_dir = osp.join(root, dataset_name)
        logger.info(f"Loading {dataset_name} from {self.dataset_dir}")        

        # self._check_before_run()
        
        def train_test_split(group):
            if len(group) < 10:
                return None
            if '999' in group.person.values[0]:
                return None
            
            # put 5% of sessions into the test set
            # if hash(group.person.values[0]) % 20 == 0:
            # print(zlib.adler32(group.person.values[0].encode()))
            if zlib.adler32(group.person.values[0].encode()) % 10 == 0:
                group = group.sample(frac=0.25).copy()
                n_records = len(group)
                n_gallery = math.floor(n_records * 0.6)
                n_train = 0
                n_query = n_records - n_gallery - n_train
                labels = ['train'] * n_train + ['gallery'] * n_gallery + ['query'] * n_query 
                group['split'] = labels
            else:
                group = group.sample(frac=0.25).copy()
                group['split'] = 'train'

            return group

        files = sorted(glob.glob(f'{self.dataset_dir}/*/*/*/*/person*.jpg'))
        if not files:
            files = sorted(glob.glob(f'{self.dataset_dir}/*/*/*/*/*/*/person*.jpg'))

        data = pd.DataFrame(files, columns=['img_path'])
        data['wave'] = data.img_path.apply(lambda x: x.split('/')[-2])
        data['wave_id'] = pd.factorize(data['wave'])[0]
        data['person'] = data.img_path.apply(lambda x: x.split('/')[-3])
        data['session'] = data.img_path.apply(lambda x: '_'.join(x.split('/')[-7:-3]))

        group_data = pd.concat([train_test_split(group) for _, group in data.groupby('wave_id')])
        # print(pd.factorize(group_data['session'])[0])
        # print(group_data['session'].unique())
        group_data['dsetid'] = pd.factorize(group_data['session'])[0]
        group_data['wave_id'] = pd.factorize(group_data['wave'])[0]    

        train_data = group_data.query('split == "train"')
        train_data['person_id'] = pd.factorize(train_data['person'])[0]
        query_data = group_data.query('split == "query"')
        query_data['person_id'] = pd.factorize(query_data['person'])[0]
        gallery_data = group_data.query('split == "gallery"')
        gallery_data['person_id'] = pd.factorize(gallery_data['person'])[0]

        train = list(train_data.apply(lambda x: (x.img_path, x.person_id + pid_offset, 0, x.dsetid + sid_offset), axis=1))
        query = list(query_data.apply(lambda x: (x.img_path, x.person_id + pid_offset, 0, x.dsetid + sid_offset), axis=1))
        gallery = list(gallery_data.apply(lambda x: (x.img_path, x.person_id + pid_offset, 1, x.dsetid + sid_offset), axis=1))

        if not include_train:
            # session_container = self._get_session_container(
            #     self.train_dir, limit=train_limit
            # )
            # pid_container = self._get_pid_container(self.train_dir, session_container)
            # train = self._process_dir(
            #     self.train_dir,
            #     pid_container,
            #     session_container,
            #     relabel=True,
            #     pid_offset=pid_offset,
            #     sid_offset=sid_offset,
            # )

        # else:
            train = []

        if not include_val:
            # session_container = self._get_session_container(
            #     self.query_dir, limit=val_limit
            # )
            # pid_container = self._get_pid_container(self.query_dir, session_container)
            # query = self._process_dir(
            #     self.query_dir,
            #     pid_container,
            #     session_container,
            #     relabel=False,
            #     is_query=True,
            # )
            # gallery = self._process_dir(
            #     self.gallery_dir, pid_container, session_container, relabel=False
            # )
        # else:
            query = []
            gallery = []

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print("=> FlowstateSessions loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
            self.num_train_vids,
            _,
            _,
            _,
            _,
        ) = self.get_imagedata_info(self.train)

        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
            self.num_query_vids,
            _,
            _,
            _,
            _,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
            self.num_gallery_vids,
            _,
            _,
            _,
            _,
        ) = self.get_imagedata_info(self.gallery)

    def _get_pid_container(self, dir_path, session_container, limit=None):
        glob_path = osp.join(dir_path, "*.jpg")
        img_paths = glob.glob(glob_path)
        pid_container = set()
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)
            # get pid
            pid = filename.split("__")[0]
            # session_id = filename.split("__")[0]
            # if session_id not in session_container:
            #     continue
            pid_container.add(pid)
        if limit is not None:
            random.seed(0)
            pid_container = random.sample(pid_container, min(limit, len(pid_container)))
        logger.info(f"Found {len(pid_container)} ids in {dir_path}")
        return pid_container

    def _get_session_container(self, dir_path, limit=None):
        glob_path = osp.join(dir_path, "*.jpg")
        img_paths = glob.glob(glob_path)
        session_container = set()
        for img_path in sorted(img_paths):
            # get filename
            filename = osp.basename(img_path)
            # get pid
            session_id = 0
            session_container.add(session_id)
        if limit is not None:
            session_container = sorted(list(session_container))[:limit]
        logger.info(f"Found {len(session_container)} sessions in {dir_path}")
        return session_container

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

    
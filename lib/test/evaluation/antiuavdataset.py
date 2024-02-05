import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class AntiUAVDataset(BaseDataset):
    """ OTB-2015 dataset
    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf
    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.antiuav_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uav', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "20190925_111757_1_1", "path": "20190925_111757_1_1/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_1/visible.txt", "object_class": "person"},
            {"name": "20190925_111757_1_10", "path": "20190925_111757_1_10/visible", "startFrame": 0, "endFrame": 347,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_10/visible.txt", "object_class": "person"},
            {"name": "20190925_111757_1_2", "path": "20190925_111757_1_2/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_2/visible.txt", "object_class": "person"},
            {"name": "20190925_111757_1_3", "path": "20190925_111757_1_3/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_3/visible.txt", "object_class": "person"},
            {"name": "20190925_111757_1_4", "path": "20190925_111757_1_4/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_4/visible.txt", "object_class": "person"},
            {"name": "20190925_111757_1_5", "path": "20190925_111757_1_5/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_5/visible.txt", "object_class": "person"},
            {"name": "20190925_111757_1_6", "path": "20190925_111757_1_6/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_6/visible.txt", "object_class": "person"},
            {"name": "20190925_111757_1_7", "path": "20190925_111757_1_7/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_7/visible.txt", "object_class": "person"},
            {"name": "20190925_111757_1_8", "path": "20190925_111757_1_8/visible", "startFrame": 56, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_8/visible.txt", "object_class": "person"},
            {"name": "20190925_111757_1_9", "path": "20190925_111757_1_9/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_111757_1_9/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_1", "path": "20190925_124000_1_1/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_1/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_10", "path": "20190925_124000_1_10/visible", "startFrame": 0, "endFrame": 163,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_10/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_2", "path": "20190925_124000_1_2/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_2/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_3", "path": "20190925_124000_1_3/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_3/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_4", "path": "20190925_124000_1_4/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_4/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_5", "path": "20190925_124000_1_5/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_5/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_6", "path": "20190925_124000_1_6/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_6/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_7", "path": "20190925_124000_1_7/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_7/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_8", "path": "20190925_124000_1_8/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_8/visible.txt", "object_class": "person"},
            {"name": "20190925_124000_1_9", "path": "20190925_124000_1_9/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124000_1_9/visible.txt", "object_class": "person"},
            {"name": "20190925_124612_1_1", "path": "20190925_124612_1_1/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124612_1_1/visible.txt", "object_class": "person"},
            {"name": "20190925_124612_1_2", "path": "20190925_124612_1_2/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124612_1_2/visible.txt", "object_class": "person"},
            {"name": "20190925_124612_1_3", "path": "20190925_124612_1_3/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124612_1_3/visible.txt", "object_class": "person"},
            {"name": "20190925_124612_1_4", "path": "20190925_124612_1_4/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124612_1_4/visible.txt", "object_class": "person"},
            {"name": "20190925_124612_1_5", "path": "20190925_124612_1_5/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124612_1_5/visible.txt", "object_class": "person"},
            {"name": "20190925_124612_1_6", "path": "20190925_124612_1_6/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124612_1_6/visible.txt", "object_class": "person"},
            {"name": "20190925_124612_1_7", "path": "20190925_124612_1_7/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124612_1_7/visible.txt", "object_class": "person"},
            {"name": "20190925_124612_1_8", "path": "20190925_124612_1_8/visible", "startFrame": 0, "endFrame": 922,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_124612_1_8/visible.txt", "object_class": "person"},
            {"name": "20190925_134301_1_1", "path": "20190925_134301_1_1/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_134301_1_1/visible.txt", "object_class": "person"},
            {"name": "20190925_134301_1_2", "path": "20190925_134301_1_2/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_134301_1_2/visible.txt", "object_class": "person"},
            {"name": "20190925_134301_1_3", "path": "20190925_134301_1_3/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_134301_1_3/visible.txt", "object_class": "person"},
            {"name": "20190925_134301_1_4", "path": "20190925_134301_1_4/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_134301_1_4/visible.txt", "object_class": "person"},
            {"name": "20190925_134301_1_5", "path": "20190925_134301_1_5/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_134301_1_5/visible.txt", "object_class": "person"},
            {"name": "20190925_134301_1_6", "path": "20190925_134301_1_6/visible", "startFrame": 483, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_134301_1_6/visible.txt", "object_class": "person"},
            {"name": "20190925_134301_1_7", "path": "20190925_134301_1_7/visible", "startFrame": 776, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_134301_1_7/visible.txt", "object_class": "person"},
            {"name": "20190925_134301_1_8", "path": "20190925_134301_1_8/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_134301_1_8/visible.txt", "object_class": "person"},
            {"name": "20190925_134301_1_9", "path": "20190925_134301_1_9/visible", "startFrame": 0, "endFrame": 618,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_134301_1_9/visible.txt", "object_class": "person"},
            {"name": "20190925_193610_1_1", "path": "20190925_193610_1_1/visible", "startFrame": 0, "endFrame": 932,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_193610_1_1/visible.txt", "object_class": "person"},
            {"name": "20190925_193610_1_2", "path": "20190925_193610_1_2/visible", "startFrame": 0, "endFrame": 931,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_193610_1_2/visible.txt", "object_class": "person"},
            {"name": "20190925_193610_1_3", "path": "20190925_193610_1_3/visible", "startFrame": 0, "endFrame": 949,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_193610_1_3/visible.txt", "object_class": "person"},
            {"name": "20190925_193610_1_4", "path": "20190925_193610_1_4/visible", "startFrame": 0, "endFrame": 942,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_193610_1_4/visible.txt", "object_class": "person"},
            {"name": "20190925_193610_1_5", "path": "20190925_193610_1_5/visible", "startFrame": 0, "endFrame": 947,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_193610_1_5/visible.txt", "object_class": "person"},
            {"name": "20190925_193610_1_6", "path": "20190925_193610_1_6/visible", "startFrame": 0, "endFrame": 922,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_193610_1_6/visible.txt", "object_class": "person"},
            {"name": "20190925_193610_1_7", "path": "20190925_193610_1_7/visible", "startFrame": 0, "endFrame": 938,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_193610_1_7/visible.txt", "object_class": "person"},
            {"name": "20190925_193610_1_8", "path": "20190925_193610_1_8/visible", "startFrame": 0, "endFrame": 920,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_193610_1_8/visible.txt", "object_class": "person"},
            {"name": "20190925_193610_1_9", "path": "20190925_193610_1_9/visible", "startFrame": 0, "endFrame": 888,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_193610_1_9/visible.txt", "object_class": "person"},
            {"name": "20190925_200805_1_1", "path": "20190925_200805_1_1/visible", "startFrame": 0, "endFrame": 949,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_200805_1_1/visible.txt", "object_class": "person"},
            {"name": "20190925_200805_1_2", "path": "20190925_200805_1_2/visible", "startFrame": 0, "endFrame": 933,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_200805_1_2/visible.txt", "object_class": "person"},
            {"name": "20190925_200805_1_3", "path": "20190925_200805_1_3/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_200805_1_3/visible.txt", "object_class": "person"},
            {"name": "20190925_200805_1_4", "path": "20190925_200805_1_4/visible", "startFrame": 0, "endFrame": 939,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_200805_1_4/visible.txt", "object_class": "person"},
            {"name": "20190925_200805_1_5", "path": "20190925_200805_1_5/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_200805_1_5/visible.txt", "object_class": "person"},
            {"name": "20190925_200805_1_6", "path": "20190925_200805_1_6/visible", "startFrame": 0, "endFrame": 861,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_200805_1_6/visible.txt", "object_class": "person"},
            {"name": "20190925_200805_1_7", "path": "20190925_200805_1_7/visible", "startFrame": 0, "endFrame": 957,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_200805_1_7/visible.txt", "object_class": "person"},
            {"name": "20190925_200805_1_8", "path": "20190925_200805_1_8/visible", "startFrame": 0, "endFrame": 932,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_200805_1_8/visible.txt", "object_class": "person"},
            {"name": "20190925_200805_1_9", "path": "20190925_200805_1_9/visible", "startFrame": 0, "endFrame": 474,
             "nz": 4, "ext": "jpg", "anno_path": "20190925_200805_1_9/visible.txt", "object_class": "person"},
            {"name": "20190926_095902_1_1", "path": "20190926_095902_1_1/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_095902_1_1/visible.txt", "object_class": "person"},
            {"name": "20190926_095902_1_2", "path": "20190926_095902_1_2/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_095902_1_2/visible.txt", "object_class": "person"},
            {"name": "20190926_095902_1_3", "path": "20190926_095902_1_3/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_095902_1_3/visible.txt", "object_class": "person"},
            {"name": "20190926_095902_1_4", "path": "20190926_095902_1_4/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_095902_1_4/visible.txt", "object_class": "person"},
            {"name": "20190926_095902_1_5", "path": "20190926_095902_1_5/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_095902_1_5/visible.txt", "object_class": "person"},
            {"name": "20190926_095902_1_6", "path": "20190926_095902_1_6/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_095902_1_6/visible.txt", "object_class": "person"},
            {"name": "20190926_095902_1_7", "path": "20190926_095902_1_7/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_095902_1_7/visible.txt", "object_class": "person"},
            {"name": "20190926_095902_1_8", "path": "20190926_095902_1_8/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_095902_1_8/visible.txt", "object_class": "person"},
            {"name": "20190926_095902_1_9", "path": "20190926_095902_1_9/visible", "startFrame": 0, "endFrame": 763,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_095902_1_9/visible.txt", "object_class": "person"},
            {"name": "20190926_102042_1_1", "path": "20190926_102042_1_1/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_102042_1_1/visible.txt", "object_class": "person"},
            {"name": "20190926_102042_1_2", "path": "20190926_102042_1_2/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_102042_1_2/visible.txt", "object_class": "person"},
            {"name": "20190926_102042_1_3", "path": "20190926_102042_1_3/visible", "startFrame": 108, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_102042_1_3/visible.txt", "object_class": "person"},
            {"name": "20190926_102042_1_4", "path": "20190926_102042_1_4/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_102042_1_4/visible.txt", "object_class": "person"},
            {"name": "20190926_102042_1_5", "path": "20190926_102042_1_5/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_102042_1_5/visible.txt", "object_class": "person"},
            {"name": "20190926_102042_1_6", "path": "20190926_102042_1_6/visible", "startFrame": 45, "endFrame": 969,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_102042_1_6/visible.txt", "object_class": "person"},
            {"name": "20190926_102042_1_7", "path": "20190926_102042_1_7/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_102042_1_7/visible.txt", "object_class": "person"},
            {"name": "20190926_102042_1_8", "path": "20190926_102042_1_8/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_102042_1_8/visible.txt", "object_class": "person"},
            {"name": "20190926_102042_1_9", "path": "20190926_102042_1_9/visible", "startFrame": 0, "endFrame": 428,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_102042_1_9/visible.txt", "object_class": "person"},
            {"name": "20190926_111509_1_1", "path": "20190926_111509_1_1/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_111509_1_1/visible.txt", "object_class": "person"},
            {"name": "20190926_111509_1_2", "path": "20190926_111509_1_2/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_111509_1_2/visible.txt", "object_class": "person"},
            {"name": "20190926_111509_1_3", "path": "20190926_111509_1_3/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_111509_1_3/visible.txt", "object_class": "person"},
            {"name": "20190926_111509_1_4", "path": "20190926_111509_1_4/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_111509_1_4/visible.txt", "object_class": "person"},
            {"name": "20190926_111509_1_5", "path": "20190926_111509_1_5/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_111509_1_5/visible.txt", "object_class": "person"},
            {"name": "20190926_111509_1_6", "path": "20190926_111509_1_6/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_111509_1_6/visible.txt", "object_class": "person"},
            {"name": "20190926_111509_1_7", "path": "20190926_111509_1_7/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_111509_1_7/visible.txt", "object_class": "person"},
            {"name": "20190926_111509_1_8", "path": "20190926_111509_1_8/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_111509_1_8/visible.txt", "object_class": "person"},
            {"name": "20190926_111509_1_9", "path": "20190926_111509_1_9/visible", "startFrame": 0, "endFrame": 138,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_111509_1_9/visible.txt", "object_class": "person"},
            {"name": "20190926_134054_1_1", "path": "20190926_134054_1_1/visible", "startFrame": 150, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_134054_1_1/visible.txt", "object_class": "person"},
            {"name": "20190926_134054_1_2", "path": "20190926_134054_1_2/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_134054_1_2/visible.txt", "object_class": "person"},
            {"name": "20190926_134054_1_3", "path": "20190926_134054_1_3/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_134054_1_3/visible.txt", "object_class": "person"},
            {"name": "20190926_134054_1_4", "path": "20190926_134054_1_4/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_134054_1_4/visible.txt", "object_class": "person"},
            {"name": "20190926_134054_1_5", "path": "20190926_134054_1_5/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_134054_1_5/visible.txt", "object_class": "person"},
            {"name": "20190926_134054_1_6", "path": "20190926_134054_1_6/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_134054_1_6/visible.txt", "object_class": "person"},
            {"name": "20190926_134054_1_7", "path": "20190926_134054_1_7/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_134054_1_7/visible.txt", "object_class": "person"},
            {"name": "20190926_134054_1_8", "path": "20190926_134054_1_8/visible", "startFrame": 0, "endFrame": 999,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_134054_1_8/visible.txt", "object_class": "person"},
            {"name": "20190926_134054_1_9", "path": "20190926_134054_1_9/visible", "startFrame": 0, "endFrame": 557,
             "nz": 4, "ext": "jpg", "anno_path": "20190926_134054_1_9/visible.txt", "object_class": "person"}
        ]
    
        return sequence_info_list
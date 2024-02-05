import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class AntiUAVforjsonDataset(BaseDataset):
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
        self.base_path = self.env_settings.antiuavforjson_path
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
            {"name": "video_0001", "path": "video_0001/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0001/anno.txt", "object_class": "person"},
            {"name": "video_0002", "path": "video_0002/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0002/anno.txt", "object_class": "person"},
            {"name": "video_0003", "path": "video_0003/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0003/anno.txt", "object_class": "person"},
            {"name": "video_0004", "path": "video_0004/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0004/anno.txt", "object_class": "person"},
            {"name": "video_0005", "path": "video_0005/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0005/anno.txt", "object_class": "person"},
            {"name": "video_0006", "path": "video_0006/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0006/anno.txt", "object_class": "person"},
            {"name": "video_0007", "path": "video_0007/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0007/anno.txt", "object_class": "person"},
            {"name": "video_0008", "path": "video_0008/img", "startFrame": 0, "endFrame": 763, "nz": 4, "ext": "jpg",
             "anno_path": "video_0008/anno.txt", "object_class": "person"},
            {"name": "video_0009", "path": "video_0009/img", "startFrame": 0, "endFrame": 347, "nz": 4, "ext": "jpg",
             "anno_path": "video_0009/anno.txt", "object_class": "person"},
            {"name": "video_0010", "path": "video_0010/img", "startFrame": 0, "endFrame": 922, "nz": 4, "ext": "jpg",
             "anno_path": "video_0010/anno.txt", "object_class": "person"},
            {"name": "video_0011", "path": "video_0011/img", "startFrame": 0, "endFrame": 932, "nz": 4, "ext": "jpg",
             "anno_path": "video_0011/anno.txt", "object_class": "person"},
            {"name": "video_0012", "path": "video_0012/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0012/anno.txt", "object_class": "person"},
            {"name": "video_0013", "path": "video_0013/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0013/anno.txt", "object_class": "person"},
            {"name": "video_0014", "path": "video_0014/img", "startFrame": 56, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0014/anno.txt", "object_class": "person"},
            {"name": "video_0015", "path": "video_0015/img", "startFrame": 776, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0015/anno.txt", "object_class": "person"},
            {"name": "video_0016", "path": "video_0016/img", "startFrame": 0, "endFrame": 949, "nz": 4, "ext": "jpg",
             "anno_path": "video_0016/anno.txt", "object_class": "person"},
            {"name": "video_0017", "path": "video_0017/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0017/anno.txt", "object_class": "person"},
            {"name": "video_0018", "path": "video_0018/img", "startFrame": 108, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0018/anno.txt", "object_class": "person"},
            {"name": "video_0019", "path": "video_0019/img", "startFrame": 0, "endFrame": 474, "nz": 4, "ext": "jpg",
             "anno_path": "video_0019/anno.txt", "object_class": "person"},
            {"name": "video_0020", "path": "video_0020/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0020/anno.txt", "object_class": "person"},
            {"name": "video_0021", "path": "video_0021/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0021/anno.txt", "object_class": "person"},
            {"name": "video_0022", "path": "video_0022/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0022/anno.txt", "object_class": "person"},
            {"name": "video_0023", "path": "video_0023/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0023/anno.txt", "object_class": "person"},
            {"name": "video_0024", "path": "video_0024/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0024/anno.txt", "object_class": "person"},
            {"name": "video_0025", "path": "video_0025/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0025/anno.txt", "object_class": "person"},
            {"name": "video_0026", "path": "video_0026/img", "startFrame": 0, "endFrame": 947, "nz": 4, "ext": "jpg",
             "anno_path": "video_0026/anno.txt", "object_class": "person"},
            {"name": "video_0027", "path": "video_0027/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0027/anno.txt", "object_class": "person"},
            {"name": "video_0028", "path": "video_0028/img", "startFrame": 0, "endFrame": 939, "nz": 4, "ext": "jpg",
             "anno_path": "video_0028/anno.txt", "object_class": "person"},
            {"name": "video_0029", "path": "video_0029/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0029/anno.txt", "object_class": "person"},
            {"name": "video_0030", "path": "video_0030/img", "startFrame": 0, "endFrame": 861, "nz": 4, "ext": "jpg",
             "anno_path": "video_0030/anno.txt", "object_class": "person"},
            {"name": "video_0031", "path": "video_0031/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0031/anno.txt", "object_class": "person"},
            {"name": "video_0032", "path": "video_0032/img", "startFrame": 0, "endFrame": 428, "nz": 4, "ext": "jpg",
             "anno_path": "video_0032/anno.txt", "object_class": "person"},
            {"name": "video_0033", "path": "video_0033/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0033/anno.txt", "object_class": "person"},
            {"name": "video_0034", "path": "video_0034/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0034/anno.txt", "object_class": "person"},
            {"name": "video_0035", "path": "video_0035/img", "startFrame": 0, "endFrame": 933, "nz": 4, "ext": "jpg",
             "anno_path": "video_0035/anno.txt", "object_class": "person"},
            {"name": "video_0036", "path": "video_0036/img", "startFrame": 0, "endFrame": 557, "nz": 4, "ext": "jpg",
             "anno_path": "video_0036/anno.txt", "object_class": "person"},
            {"name": "video_0037", "path": "video_0037/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0037/anno.txt", "object_class": "person"},
            {"name": "video_0038", "path": "video_0038/img", "startFrame": 45, "endFrame": 969, "nz": 4, "ext": "jpg",
             "anno_path": "video_0038/anno.txt", "object_class": "person"},
            {"name": "video_0039", "path": "video_0039/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0039/anno.txt", "object_class": "person"},
            {"name": "video_0040", "path": "video_0040/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0040/anno.txt", "object_class": "person"},
            {"name": "video_0041", "path": "video_0041/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0041/anno.txt", "object_class": "person"},
            {"name": "video_0042", "path": "video_0042/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0042/anno.txt", "object_class": "person"},
            {"name": "video_0043", "path": "video_0043/img", "startFrame": 0, "endFrame": 949, "nz": 4, "ext": "jpg",
             "anno_path": "video_0043/anno.txt", "object_class": "person"},
            {"name": "video_0044", "path": "video_0044/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0044/anno.txt", "object_class": "person"},
            {"name": "video_0045", "path": "video_0045/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0045/anno.txt", "object_class": "person"},
            {"name": "video_0046", "path": "video_0046/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0046/anno.txt", "object_class": "person"},
            {"name": "video_0047", "path": "video_0047/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0047/anno.txt", "object_class": "person"},
            {"name": "video_0048", "path": "video_0048/img", "startFrame": 0, "endFrame": 957, "nz": 4, "ext": "jpg",
             "anno_path": "video_0048/anno.txt", "object_class": "person"},
            {"name": "video_0049", "path": "video_0049/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0049/anno.txt", "object_class": "person"},
            {"name": "video_0050", "path": "video_0050/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0050/anno.txt", "object_class": "person"},
            {"name": "video_0051", "path": "video_0051/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0051/anno.txt", "object_class": "person"},
            {"name": "video_0052", "path": "video_0052/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0052/anno.txt", "object_class": "person"},
            {"name": "video_0053", "path": "video_0053/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0053/anno.txt", "object_class": "person"},
            {"name": "video_0054", "path": "video_0054/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0054/anno.txt", "object_class": "person"},
            {"name": "video_0055", "path": "video_0055/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0055/anno.txt", "object_class": "person"},
            {"name": "video_0056", "path": "video_0056/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0056/anno.txt", "object_class": "person"},
            {"name": "video_0057", "path": "video_0057/img", "startFrame": 0, "endFrame": 942, "nz": 4, "ext": "jpg",
             "anno_path": "video_0057/anno.txt", "object_class": "person"},
            {"name": "video_0058", "path": "video_0058/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0058/anno.txt", "object_class": "person"},
            {"name": "video_0059", "path": "video_0059/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0059/anno.txt", "object_class": "person"},
            {"name": "video_0060", "path": "video_0060/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0060/anno.txt", "object_class": "person"},
            {"name": "video_0061", "path": "video_0061/img", "startFrame": 0, "endFrame": 920, "nz": 4, "ext": "jpg",
             "anno_path": "video_0061/anno.txt", "object_class": "person"},
            {"name": "video_0062", "path": "video_0062/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0062/anno.txt", "object_class": "person"},
            {"name": "video_0063", "path": "video_0063/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0063/anno.txt", "object_class": "person"},
            {"name": "video_0064", "path": "video_0064/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0064/anno.txt", "object_class": "person"},
            {"name": "video_0065", "path": "video_0065/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0065/anno.txt", "object_class": "person"},
            {"name": "video_0066", "path": "video_0066/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0066/anno.txt", "object_class": "person"},
            {"name": "video_0067", "path": "video_0067/img", "startFrame": 483, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0067/anno.txt", "object_class": "person"},
            {"name": "video_0068", "path": "video_0068/img", "startFrame": 0, "endFrame": 931, "nz": 4, "ext": "jpg",
             "anno_path": "video_0068/anno.txt", "object_class": "person"},
            {"name": "video_0069", "path": "video_0069/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0069/anno.txt", "object_class": "person"},
            {"name": "video_0070", "path": "video_0070/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0070/anno.txt", "object_class": "person"},
            {"name": "video_0071", "path": "video_0071/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0071/anno.txt", "object_class": "person"},
            {"name": "video_0072", "path": "video_0072/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0072/anno.txt", "object_class": "person"},
            {"name": "video_0073", "path": "video_0073/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0073/anno.txt", "object_class": "person"},
            {"name": "video_0074", "path": "video_0074/img", "startFrame": 0, "endFrame": 932, "nz": 4, "ext": "jpg",
             "anno_path": "video_0074/anno.txt", "object_class": "person"},
            {"name": "video_0075", "path": "video_0075/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0075/anno.txt", "object_class": "person"},
            {"name": "video_0076", "path": "video_0076/img", "startFrame": 0, "endFrame": 138, "nz": 4, "ext": "jpg",
             "anno_path": "video_0076/anno.txt", "object_class": "person"},
            {"name": "video_0077", "path": "video_0077/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0077/anno.txt", "object_class": "person"},
            {"name": "video_0078", "path": "video_0078/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0078/anno.txt", "object_class": "person"},
            {"name": "video_0079", "path": "video_0079/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0079/anno.txt", "object_class": "person"},
            {"name": "video_0080", "path": "video_0080/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0080/anno.txt", "object_class": "person"},
            {"name": "video_0081", "path": "video_0081/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0081/anno.txt", "object_class": "person"},
            {"name": "video_0082", "path": "video_0082/img", "startFrame": 0, "endFrame": 922, "nz": 4, "ext": "jpg",
             "anno_path": "video_0082/anno.txt", "object_class": "person"},
            {"name": "video_0083", "path": "video_0083/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0083/anno.txt", "object_class": "person"},
            {"name": "video_0084", "path": "video_0084/img", "startFrame": 0, "endFrame": 888, "nz": 4, "ext": "jpg",
             "anno_path": "video_0084/anno.txt", "object_class": "person"},
            {"name": "video_0085", "path": "video_0085/img", "startFrame": 0, "endFrame": 618, "nz": 4, "ext": "jpg",
             "anno_path": "video_0085/anno.txt", "object_class": "person"},
            {"name": "video_0086", "path": "video_0086/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0086/anno.txt", "object_class": "person"},
            {"name": "video_0087", "path": "video_0087/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0087/anno.txt", "object_class": "person"},
            {"name": "video_0088", "path": "video_0088/img", "startFrame": 0, "endFrame": 163, "nz": 4, "ext": "jpg",
             "anno_path": "video_0088/anno.txt", "object_class": "person"},
            {"name": "video_0089", "path": "video_0089/img", "startFrame": 0, "endFrame": 938, "nz": 4, "ext": "jpg",
             "anno_path": "video_0089/anno.txt", "object_class": "person"},
            {"name": "video_0090", "path": "video_0090/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0090/anno.txt", "object_class": "person"},
            {"name": "video_0091", "path": "video_0091/img", "startFrame": 0, "endFrame": 999, "nz": 4, "ext": "jpg",
             "anno_path": "video_0091/anno.txt", "object_class": "person"},
        ]
    
        return sequence_info_list
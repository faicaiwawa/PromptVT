from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/qiuyang/PromptVT/data/got10k_lmdb'
    settings.got10k_path = '/home/qiuyang/PromptVT/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/qiuyang/PromptVT/data/lasot_lmdb'
    settings.lasot_path = '/home/qiuyang/PromptVT/data/lasot'
    settings.network_path = '/home/qiuyang/PromptVT/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/qiuyang/PromptVT/data/nfs'
    settings.otb_path = '/home/qiuyang/PromptVT/data/OTB2015'
    settings.prj_dir = '/home/qiuyang/PromptVT'
    settings.result_plot_path = '/home/qiuyang/PromptVT/test/result_plots'
    settings.results_path = '/home/qiuyang/PromptVT/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/qiuyang/PromptVT'
    settings.segmentation_path = '/home/qiuyang/PromptVT/test/segmentation_results'
    settings.tc128_path = '/home/qiuyang/PromptVT/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/qiuyang/PromptVT/data/trackingNet'
    settings.uav_path = '/home/qiuyang/datasets/UAV123'
    settings.vot_path = '/home/qiuyang/PromptVT/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings


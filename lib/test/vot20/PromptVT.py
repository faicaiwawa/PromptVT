import sys
sys.path.append('/home/qiuyang/PromptVT')
from lib.test.vot20.PromptVT_vot20 import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
run_vot_exp('PromptVT', 'baseline', vis=False)

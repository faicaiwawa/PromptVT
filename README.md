# PromptVT
This project is the implementation of paper [PromptVT](https://ieeexplore.ieee.org/document/10468656), including **models**, **raw results**, and **testing codes**. 

Due to a failure of the school server, the **training code was lost**. The training framework of this model is basically the same as that of STARK-ST, so you can refer to its implementation.

This is the CPU edition, no CUDA or GPU required.

## Performance
PromptVT  achieves SOTA performance on 8 benchmarks in lightweight trackers.
![图片1](external/vot20/PromptVT/0.png)


<img src="external/vot20/PromptVT/1.png" alt="图片2" width="400"> <img src="external/vot20/PromptVT/6.png" alt="图片2" width="400">

![图片1](external/vot20/PromptVT/2.png)


<img src="external/vot20/PromptVT/4.png" alt="图片2" width="400"> <img src="external/vot20/PromptVT/5.png" alt="图片2" width="400">
<img src="external/vot20/PromptVT/3.png" alt="图片2" width="400">


## Usage
### Installation
Create and activate a conda environment:
```
conda create -n PromptVT python=3.7
conda activate PromptVT
```
Install the required packages:
```
bash install_PromptVT.sh
```

### Data Preparation
```
${PromptVT_ROOT}
 -- data
     -- lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- got10k
         |-- test
         |-- train
         |-- val
     -- OTB100
         |-- Basketball
         |-- Biker
         ...
     -- trackingnet
         |-- TRAIN_0
         |-- TRAIN_1
         ...
         |-- TRAIN_11
         |-- TEST
     -- uav123
         |-- anno
              |-- UAV123
         |-- data_seq
              |-- UAV123
     -- Anti-UAV
         |-- Test
              |-- 20190925_111757_1_1
              ...
     -- Anti-UAV-410
         |-- Test
              |-- 02_6319_1500-2999
              ...
         
```
### Path Setting
Run the following command to set paths:
```
cd < PATH_of_PromptVT >
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```
## Test and evaluate PromptVT on benchmarks
If you want to use ONNX model, set ' use_onnx = True ' in `./lib/test/tracker/PromptVT.py`.

* LaSOT
  ```
  python tracking/test.py --tracker_name PromptVT --tracker_param baseline --dataset lasot
  python tracking/analysis_results.py # need to modify tracker configs and names
  ```
* GOT10K-test
  ```
  python tracking/test.py --tracker_name PromptVT --tracker_param baseline --dataset got10k_test
  python lib/test/utils/transform_got10k.py --tracker_name PromptVT --cfg_name baseline
  ```
  Upload the results to the official [GOT-10K evaluation server](http://got-10k.aitestunion.com/).
  
* TrackingNet
  ```
  python tracking/test.py --tracker_name PromptVT --tracker_param baseline --dataset trackingnet
  python lib/test/utils/transform_trackingnet.py --tracker_name PromptVT --cfg_name baseline
  ```
  Upload the results to the official [TrackingNet evaluation server](https://eval.ai/web/challenges/challenge-page/1805/overview).
  
* UAV123
  ```
  python tracking/test.py --tracker_name PromptVT --tracker_param baseline --dataset uav
  python tracking/analysis_results.py # need to modify tracker configs and names
  ```
* AntiUAV
  ```
  python tracking/test.py --tracker_name PromptVT --tracker_param baseline --dataset antiuav
  python tracking/analysis_results.py # need to modify tracker configs and names
  ```
  The raw data is labeled in json format, which we converted to OTB-like-txt format to fit our tracking library. The converted file is located at `. /tracking/AntiUAVJSON2OTBTxt.py`.
* AntiUAV410
  ```
  python tracking/test.py --tracker_name PromptVT --tracker_param baseline --dataset antiuav410
  python tracking/analysis_results.py # need to modify tracker configs and names
  ```
  The raw data is labeled in json format, which we converted to OTB-like-txt format to fit our tracking library. The converted file is located at `. /tracking/AntiUAVJSON2OTBTxt.py`.
* OTB100
  ```
  python tracking/test.py --tracker_name PromptVT --tracker_param baseline --dataset otb
  python tracking/analysis_results.py # need to modify tracker configs and names
  ```
* VOT2020  
  modify the path sets in `./external/vot20/trackers.ini`, `./lib/test/vot20/PromptVT.py`, and `./lib/test/vot20/PromptVT_vot20.py`.
 
   ```
  cd external/vot20/PromptVT
  bash exp.sh
    ```
  
## Test FLOPs, Params, and FPS
  ####  FLOPs and Params:
  modify the ' yaml_fname ' in `./tracking/profile_model.py`.
   ```
  python tracking/profile_model.py
  ```
   ####  FPS:
  place the `tracking/Calculate_FPS.py` in the tracking results folder and run it.
  
##  Model Zoo & Raw Results
The trained models and the raw tracking results are provided in the [model zoo](https://drive.google.com/file/d/1QMk1rlOhQWsztc7fYUOLd5sW08OC1_-Z/view?usp=drive_link).<br />
put PyTorch model and  ONNX model in  `./checkpoints/PromptVT/baseline/`.<br />
We also provide model conversion scripts`./tracking/****_onnx.py`.


## Acknowledgments
Thanks for the [PyTracking](https://github.com/visionml/pytracking) and [STARK](https://github.com/researchmm/Stark) for helping us quickly implement our ideas.

## Contact
If you have any question, feel free to email qiuyangzhang2022@163.com. ^_^



  


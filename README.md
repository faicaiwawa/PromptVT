# PromptVT
This project is the implementation of paper PromptVT, including models and testing codes(the training code will be uploaded after organizing).

:exclamation: Ubuntu(Linux) is highly recommended, Windows has some weird installation problems and model inference problems.

:exclamation: This is the CPU edition, no CUDA or GPU required.

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
The trained models and the raw tracking results are provided in the [model zoo](https://drive.google.com/file/d/1AuYS8OUXbHhe7QlfVF9vC3EVyzNsReDC/view?usp=drive_link).<br />
put pytorch model in  `./checkpoints/PromptVT/baseline/`, put onnx model in `./tracking/`.<br />
We also provide model conversion scripts . `./tracking/****_onnx.py`
 

## Acknowledgments
Thanks for the [PyTracking](https://github.com/visionml/pytracking) and [STARK](https://github.com/researchmm/Stark) for helping us quickly implement our ideas.



  


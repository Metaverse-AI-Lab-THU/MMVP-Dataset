# MMVP
<img src="docs/teaser.png" width="100%">

This repo is used for optimzation and visualization for MMVP dataset:

MMVP: A Multimodal MoCap Dataset with Vision and Pressure Sensors  
[He Zhang](https://github.com/zhanghebuaa), [Shenghao Ren](https://www.wjrzm.com/), [Haolei Yuan](https://github.com/haolyuan),
Jianhui Zhao, Fan Li, Shuangpeng Sun, Zhenghao Liang, [Tao Yu](https://ytrock.com/), Qiu Shen, Xun Cao

# Overview
<img src="docs/MoCap_20230422_172438.gif" width="50%"><img src="docs/MoCap_20230422_132043.gif" width="50%"><img src="docs/MoCap_20230422_151220.gif" width="50%"><img src="docs/MoCap_20230422_182056.gif" width="50%">

# agreement
1. The MMVP dataset (the "Dataset") is available for non-commercial research purposes only. Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, as training data for a commercial product, for commercial ergonomic analysis (e.g. product design, architectural design, etc.), or production of other artifacts for commercial purposes including, for example, web services, movies, television programs, mobile applications, or video games. The dataset may not be used for pornographic purposes or to generate pornographic material whether commercial or not. The Dataset may not be reproduced, modified and/or made available in any form to any third party without Tsinghua University’s prior written permission.
2. You agree not to reproduce, modified, duplicate, copy, sell, trade, resell or exploit any portion of the images and any portion of derived data in any form to any third party without Tsinghua University’s prior written permission.
3. You agree not to further copy, publish or distribute any portion of the Dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.
4. Tsinghua University reserves the right to terminate your access to the Dataset at any time.

# Visualizing
Visualizing code has been released. Follow the step below for visualizing:
1. Fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSf1uLLqXfnVXR2MoyRv5bF9D5gEMc8m8YASa38sTohzVcaVUg/viewform?usp=sf_link) to request authorization to use Motion-X for non-commercial purposes. It may take one week to get reply. Or Contact [Tao Yu](https://ytrock.com/) at (ytrock@126.com).
2. MMVP Dataset are following [this structure](#datasets-structure). 
3. Users could visualize dataset with the command below:
```
sh demo.sh
```
You could change frame info in `configs/visualizing.yaml`.  
`basdir` is the root where you put MMVP Dataset.    
`dataset` is fixed as `20230422`.   
`sub_ids` could be `S01...S12`  
`seq_name` represents the seq under the `sub_ids` you select.   
Noticing that `frame_idx` may not cover all frames and could be selected in `annotations/smpl_pose`.

We provide a few frames in `debug/` for visualizing. The visualizing result will be saved in `output/`. Here is a example for visualizing result: 

<img src="output/renders_gt/20230422/S04/MoCap_20230422_135735/102.jpg" width="50%"><img src="output/insoles_gt/20230422/S04/MoCap_20230422_135735/102.jpg" width="30%">


# Datasets Structure
MMVP dataset are following the structure below:
```
images
└── 20230422
    ├── S01
    ├── ...
    └── S12
        ├── MoCap_20230422_145333
        ├── ...
        └── MoCap_20230422_150723
            ├── color
            ├── depth
            ├── depth_mask
            └── calibration.npy
annotations
└── 20230422
    ├─ floor_info
    │   ├── floor_S01.npy
    │   ├── ...
    │   └── floor_S12.npy
    │
    └─ smpl_pose
        ├── S01
        ├── ...
        └── S12
            ├── MoCap_20230422_145333
            ├── ...
            └── MoCap_20230422_150723
```
# Citation
If you find our work useful in your research, please cite our paper [MMVP](https://metaverse-ai-lab-thu.github.io/MMVP-Dataset/):
```bibtex
@inproceedings{Zhang2024MMVP,
    title={MMVP: A Multimodal MoCap Dataset with Vision and Pressure Sensors},
    author={He Zhang, Shenghao Ren, Haolei Yuan, Jianhui Zhao, Fan Li, Shuangpeng Sun, Zhenghao Liang, Tao Yu, Qiu Shen, Xun Cao},
    journal={CVPR},
    year={2024}
}
```
# Contact
- Tao Yu (ytrock@126.com)
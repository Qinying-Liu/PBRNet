This repository is the introduction of "PBRNet:Progressive Boundary Refinement Network for Temporal Action Detection"(AAAI2020) from USTC VIM Lab. They are designed for accurate and efficient temporal action detection.

## Prerequisites
python 3.6

pytorch 1.6

opencv-python 3.4.1

## Data preparation
We first download the [THUMOS14](http://crcv.ucf.edu/THUMOS14/) datasets, then sample frames from each video by 10 fps and resize each frame to the spatial size of 320x180. You can change these configurations based on your GPU resources. For optial flow extraction, we refer to [TV-L1](https://github.com/deepmind/kinetics-i3d/pull/5/files/f1fa01a332179e82cd655e7cd2f2f0c1c04f0c74) which only requires CPU. 

## Runing
```
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py
```

## Reference
```
@inproceedings{liu2020progressive,
  title={Progressive boundary refinement network for temporal action detection},
  author={Liu, Qinying and Wang, Zilei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2020}
}
```

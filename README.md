# Vision-Language to Emotion (VL2E) Model Embodied on the Humanoid Robot

## Introduction
This repo, named **HumanoidVLE**, contains the official PyTorch implementation of the system part of our paper [**UGotMe: An Embodied System for Affective Human-Robot Interaction**](https://arxiv.org/pdf/2410.18373).

Code for model training and evaluation on the MELD dataset is available here: https://github.com/lipzh5/VisionLanguageEmotion


## Getting Started

### 1. Clone the code the prepare the environment 
```
git clone git@github.com:lipzh5/HumanoidVLE.git
cd HumanoidVLE
# create env using conda for CUDA 12.1
conda create -n humanoidbackend python=3.9.19 
conda activate humanoidbackend
pip install -r requirements.txt

```

### 2. Run the server
**NOTE**: pay attention to the dynamic ip address of the robot 
```
python main.py  
```

## Disclaimer

This repository is created for research purposes and is not officially associated with Ameca or Engineered Arts. If you have any questions or encounter issues, please feel free to submit a GitHub issue.



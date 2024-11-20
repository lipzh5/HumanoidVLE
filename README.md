# Vision-Language to Emotion (VL2E) Model Embodied on the Humanoid Robot

## Introduction
This repo, named **HROVLE**, contains the official PyTorch implementation of the system part of our paper **UGotMe: An Embodied System for Affective Human-Robot Interaction**.

Code for model training and evaluation on the MELD dataset is available here: https://github.com/lipzh5/VisionLanguageEmotion


## Getting Started

### 1. Clone the code the prepare the environment 
```
git clone git@github.com:lipzh5/HROVLE.git
cd HROVLE
# create env using conda for CUDA 12.1
conda create -n hrobackend python=3.9.19 
conda activate hrobackend
pip install -r requirements.txt

```

### 2. Run the server
**NOTE**: pay attention to the dynamic ip address of the robot 
```
python main.py  
```


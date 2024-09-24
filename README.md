# Vision-Language to Emotion (VL2E) Model Embodied on Ameca

## Introduction
This repo, named **AmecaVLE**, contains the official PyTorch implementation of the system part of our paper 

code for model training and evaluation on the MELD dataset is available here: https://github.com/lipzh5/VisionLanguageEmotion


## Getting Started

### 1. Clone the code the prepare the environment 
```
git clone git@github.com:lipzh5/AmecaVLE.git
cd AmecaVLE
# create enc using conda for CUDA 12.1
conda create -n amecabackend python=3.9.19 
conda activate amecabackend
pip install -r requirements.txt

```

### 2. Run the server
**NOTE**: pay attention to the dynamic ip address of the robot 
```
python main.py  
```


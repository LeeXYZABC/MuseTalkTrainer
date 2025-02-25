# MuseTalkTrainer

The unofficial training scripts for [MuseTalk](https://github.com/TMElyralab/MuseTalk).
the model code are modified from MuseTalk, and the training and loss related code are modified from Wav2lip and Codeformer.


## Preparing
1. set hdtf_dir and hdtf_samples_dir in datasets_process.py
hdtf_dir contains the hdtf datasets and hdtf_samples_dir is the dir that will cache the training data.
2. run the data process script.
```
python datasets_process.py
```
First, Detect and crop the face region and resized to 256x256 to match the musetalk standards.
Then, use vae encoder to obtain the latents for each 256x256 region.
And, use whisper model to obtain the whisper chunks.
The 256x256 face, latents, as well as whisper chunks will be save to avoid the computing resources in training.


## Training
1. As shown in the following example, the setting of  training parameters is in musetalk_trainer.py.
```python
    hdtf_samples_dir = "/opt/data/hdtf/samples"    
    checkpoint_dir = "./checkpoints"
```
You can edit the training parameters and choose optimizer directly in the training scripts. 

2. run the training script. 
```
python musetalk_trainer.py
```
The weight parameters will be saved in log/trainer.log. 


## Download weights
You can download weights manually and place in folder models.
1. the unet wights
[unet weights](https://drive.google.com/drive/folders/1USuHLbs1ff3mFJ5QtJMAZxm_yWLwqYai?usp=sharing)

2. the weights of other components:
   [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   [whisper](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
   [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   [face-parse-bisent](https://github.com/zllrunning/face-parsing.PyTorch)
   [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)

Similar as MuseTalk, these weights should be organized in `models` as follows:
```
./models/
├── musetalk
│   └── musetalk.json
│   └── cunet.bin
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```


## inference
The inference example is shown in demo.ipynb. 


## Acknowledgements
We thank the following projects for their work that enabled us to collect the data and define the network.
1. [MuseTalk](https://github.com/TMElyralab/MuseTalk)
2. [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
3. [CodeFormer](https://github.com/sczhou/CodeFormer)

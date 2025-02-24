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


## Acknowledgements
We thank the following projects for their work that enabled us to collect the data and define the network.
1. [MuseTalk](https://github.com/TMElyralab/MuseTalk)
2. [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
3. [CodeFormer](https://github.com/sczhou/CodeFormer)

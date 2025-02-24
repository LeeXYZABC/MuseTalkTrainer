import os
import json
import cv2
import random
import pickle
import torch


def rgb2tensor(img):
    img = img[..., ::-1] # BGR to RGB
    img = img / 255.
    img = torch.FloatTensor(img) # to tensor
    img = img.permute(2, 0, 1)
    img = (img - 0.5) * 2 
    img = img.clamp(-1, 1) # make the output range the same as the vae decode
    return img

def get_pickle_content(latent_name):
    with open(latent_name, "rb") as f:
        return pickle.load(f)

def preprocess(img_names, latent_names, whisper_names):
    total_len = min(len(img_names), len(latent_names), len(whisper_names))
    idx = random.randint(0, total_len - 1)

    curr_latents = get_pickle_content(latent_names[idx])
    
    masked_latents = curr_latents[:, :4]
    latents_gt = curr_latents[:, 4:]
    latents_gt = latents_gt[0]
    
    idx_ref = random.randint(0, total_len - 1)
    while idx_ref == idx:
        idx_ref = random.randint(0, total_len - 1)
        
    ref_latents = get_pickle_content(latent_names[idx_ref])[:,4:]
    
    latents_input = torch.cat([masked_latents, ref_latents], dim=1)
    latents_input = latents_input[0]
    
    whisper_feature = get_pickle_content(whisper_names[idx])
    whisper_input = torch.from_numpy(whisper_feature)
    
    img_path = img_names[idx]
    img = cv2.imread(img_path)
    img_gt = rgb2tensor(img)
    
    return {
        "latents_gt": latents_gt, 
        "latents_input": latents_input,
        "whisper_input": whisper_input,
        "image_gt": img_gt,
    }


class TrainingDataset(object):
    def __init__(self, hdtf_dir=None, sample_repeats=1000):
        super(TrainingDataset, self).__init__()
        self.hdtf_dir = hdtf_dir
        self.sample_repeats = sample_repeats
        self.person_dirs = []
        for person in os.listdir(self.hdtf_dir):
            person_dir = os.path.join(self.hdtf_dir, person)
            splits = []
            for cut in os.listdir(person_dir):
                cut_dir = os.path.join(person_dir, cut)
                latents_file = os.path.join(cut_dir, "latents.pkl")
                whisper_file = os.path.join(cut_dir, "whisper_chunks.pkl")
                if os.path.exists(latents_file) == False or os.path.exists(whisper_file) == False:
                    print(f"latents or whisper_chunks missed", latents_file, whisper_file)
                    continue
                config_file = os.path.join(cut_dir, "config.json")
                if os.path.exists(config_file) == False:
                    print(f"config file not exists", config_file)
                    continue
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                except Exception:
                    print("config reading error", config_file)
                    continue
                if "latents" not in config or "whisper_chunks" not in config:
                    print("latents or whisper_chunks missed in config", config_file)
                    continue
                splits.append([cut_dir, latents_file, whisper_file])
            if splits:
                self.person_dirs.append(splits)

                
    def __getitem__(self, index):
        while 1:
            index = random.randint(0, len(self.person_dirs) - 1)
            splits = self.person_dirs[index]
            cut_dir, latents_file, whisper_file = random.choice(splits)
            config_file = os.path.join(cut_dir, "config.json")
            if os.path.exists(config_file) == False:
                print(f"config file not exists", config_file)
                continue
            try:
                with open(config_file) as f:
                    config = json.load(f)
            except Exception:
                print("config reading error", config_file)
                continue
            total_len = config["latents"]
            img_names = [f"{cut_dir}/{str(i)}.png" for i in range(total_len)]
            latent_names = [f"{cut_dir}/latents_{str(i)}.pkl" for i in range(config["latents"])]
            whisper_names= [f"{cut_dir}/whisper_chunks_{str(i)}.pkl" for i in range(config["whisper_chunks"])]
            return preprocess(img_names, latent_names, whisper_names)

    
    def __len__(self):
        return len(self.person_dirs) * self.sample_repeats



if __name__ == "__main__":
    tdata = TrainingDataset()
    print(len(tdata))



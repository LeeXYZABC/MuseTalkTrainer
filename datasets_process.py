
import os
import gc
import json
import copy
import pickle
import subprocess
import torch
import cv2
from tqdm import tqdm
from musetalk.utils.utils import load_all_model
from musetalk.utils.blending import get_image, get_image_prepare_material,get_image_blending
from musetalk.utils.preprocessing import get_landmark_and_bbox_from_frames, coord_placeholder

audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

hdtf_dir = "/opt/datasets/hdtf/split_video_25fps"
hdtf_samples_dir = "/opt/datasets/hdtf/samples"


template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'


def process_audio_file(vfile, afile):
    command = template.format(vfile, afile)
    print(command)
    subprocess.call(command, shell=True)


def cord_process(cords):
    new_cords = []
    for cord in cords:
        x1, y1, x2, y2 = cord
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        new_cords.append((x1, y1, x2, y2))
    return new_cords


def audio_process(work_dir="", video_file="", fps=25):
    wav_file = os.path.join(work_dir, "audio.wav")
    process_audio_file(video_file, wav_file)
    whisper_feature = audio_processor.audio2feat(wav_file)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
    
    whisper_file = os.path.join(work_dir, "whisper_chunks.pkl")
    with open(whisper_file, "wb") as f:
        pickle.dump(whisper_chunks, f)

    for i in range(len(whisper_chunks)):
        whisper_part_file = os.path.join(work_dir, f"whisper_chunks_{i}.pkl")
        with open(whisper_part_file, "wb") as f:
            pickle.dump(whisper_chunks[i], f)

    whisper_chunks_length = len(whisper_chunks)
    print(f"whisper_chunks length: {whisper_chunks_length}")
    
    del whisper_chunks
    gc.collect()

    return whisper_chunks_length


def frame_process(work_dir="", frame=None, index=-1):
    if index < 0:
        return

    bbox_shift = 0
    coord_list, frame_list = get_landmark_and_bbox_from_frames([frame], bbox_shift, tqdm_enable=False)
    coord_list = cord_process(coord_list)
    bbox = coord_list[0]
    if bbox == coord_placeholder:
        print(f"coord_placeholder :{coord_placeholder}")
        cv2.imwrite("coord_placeholder.png", frame)
        return
    
    x1, y1, x2, y2 = bbox
    crop_frame = frame[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame.copy(), (256,256), interpolation = cv2.INTER_LANCZOS4)
    latents = vae.get_latents_for_unet(crop_frame).detach().cpu()

    rgb_file = os.path.join(work_dir, f"{index}.png")
    cv2.imwrite(rgb_file, crop_frame)

    latent_part_file = os.path.join(work_dir, f"latents_{index}.pkl")
    with open(latent_part_file, "wb") as f:
        pickle.dump(latents, f)

    return bbox, latents

    
def video_process(work_dir="", video_file=""):
    print(f"now process: work_dir: {work_dir}, video_file: {video_file}")
    
    coord_list = []
    input_latent_list = []
    i = 0
    video_stream = cv2.VideoCapture(video_file)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    # audio
    whisper_chunks_length = audio_process(work_dir=work_dir, video_file=video_file, fps=fps)
    # video
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        bbox, latents = frame_process(work_dir=work_dir, frame=frame, index=i)
        coord_list.append(copy.deepcopy(bbox))
        input_latent_list.append(latents.clone())
        if i % 100 == 0:
            print(f"processed frame {i}")
        i += 1  
    coord_file = os.path.join(work_dir, "coord_list.pkl")
    with open(coord_file, "wb") as f:
        pickle.dump(coord_list, f)
    latents_file = os.path.join(work_dir, "latents.pkl")
    with open(latents_file, "wb") as f:
        pickle.dump(input_latent_list, f)    
    print(f"fps: {fps}, latents: {len(input_latent_list)}, coord_list: {len(coord_list)}")
    config = {
        "latents": len(input_latent_list),
        "whisper_chunks": whisper_chunks_length
    }
    config_file = os.path.join(work_dir, "config.json")
    print(config_file, config)
    with open(config_file, "w") as f:
        json.dump(config, f)

    del coord_list
    del input_latent_list
    gc.collect() 
    torch.cuda.empty_cache()


samples = {}

def video_run(file_path):
    filename = os.path.basename(file_path)
    person_name = ""
    for c in filename:
        if c == "_" or c.isdigit() == True or c == ".":
             break
        person_name += c
    person_dir = os.path.join(hdtf_samples_dir, person_name)
    if os.path.exists(person_dir) == False:
        os.mkdir(person_dir)
    basename = filename.split(".")[0]
    person_split_dir = os.path.join(person_dir, basename)
    if os.path.exists(person_split_dir) == False:
        os.mkdir(person_split_dir)
    else:
        print(f"skiped {person_split_dir}")
        return
    
    if person_name not in samples:
        samples[person_name] = []
    samples[person_name].append(person_split_dir)

    video_process(work_dir=person_split_dir, video_file=file_path)
    print(person_name, filename, person_dir, person_split_dir, len(samples[person_name]))


# MittRomney_2.mp4视频有问题
cnt = 0
for filename in os.listdir(hdtf_dir):
    if filename.endswith("mp4") == False:
        continue
    file_path = os.path.join(hdtf_dir, filename)
    video_run(file_path)
    cnt += 1
    print(f"now_is {cnt}")

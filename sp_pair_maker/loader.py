from diffusers import StableDiffusionPipeline
import torch
import clip
from PIL import Image
import pickle
import os


class Loader:
    def __init__(self,):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = pipe.to(device)
        self.clip, self.preprocess = clip.load("ViT-B/32", device=device)
        # Get the current directory of the script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the parent directory and then to module2
        file_path = os.path.join(os.path.dirname(current_dir), "dataset/sentence_WP", "train.pickle")
        self.sentence_wp_train=open(file_path,"rb")
        file_path = os.path.join(os.path.dirname(current_dir), "dataset/sentence_WP", "valid.pickle")
        self.sentence_wp_valid=open(file_path,"rb")
        file_path = os.path.join(os.path.dirname(current_dir), "dataset/sentence_WP", "test.pickle")
        self.sentence_wp_test=open(file_path,"rb")


    def get_stable_diffusion_pipe(self,):
        return self.pipe
    def get_clip_and_preprocess(self,):
        return self.clip, self.preprocess
    def get_sentence_wp(self,):
        return self.sentence_wp_train, self.sentence_wp_valid, self.sentence_wp_test
    def get_device(self,)
        return self.device

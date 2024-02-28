from diffusers import StableDiffusionPipeline
import torch
import clip
from PIL import Image
import pickle
import os
import languagemodels
import loader

class PipeLine():
    def __init__(self,train=True, use_GPT4=False):
        self.ld=loader.Loader()
        self.pipe=self.ld.get_stable_diffusion_pipe()
        self.clip,self.preprocess=self.ld.get_clip_and_preprocess()
        self.device=self.ld.get_device()

        self.train=train
        self.use_GPT4=use_GPT4
        

        if self.train is not False:
            if self.use_GPT4 is False:
                self.lm=languagemodels.MyLanguageModel()
            else:
                self.lm=languagemodels.OPENAI()
        else:
            self.lm=languagemodels.DefaultLM()

    def get_prompt(self,sentence,new_story):
        return self.lm.get_prompt(sentence,new_story)

    def get_image(self,prompt):
        return self.pipe(prompt).images[0]
    
    def get_logits(self,image,prompt,sentence):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = self.clip.tokenize([prompt]).to(self.device)
        
        with torch.no_grad():
            
            #image_features = clip.encode_image(image)
            #text_features = clip.encode_text(text)

            logits_per_image, logits_per_text = self.clip(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)
        return probs
    
    def lm_update(self,probs):
        self.lm.update(probs)

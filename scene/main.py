from .generator import *

def story_generation(finetune=False,gpt_modelname="gpt-3.5-turbo"):
    story_generator=Story_Generator(finetune,gpt_modelname)

if __name__ == "__main__":
    story_generation()

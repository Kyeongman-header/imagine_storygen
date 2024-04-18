from .secret import Secret
from openai import OpenAI
import pickle
import anthropic
import os
from transformers import AutoTokenizer, MistralForCausalLM

claude3_client=anthropic.Anthropic(
        api_key=Secret().get_claude3_api_key()
        )

openai_client=OpenAI(
        api_key=Secret().get_openai_api_key()
        )

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

current_dir =os.path.abspath(__file__)


def call_claude3(max_tokens=2000,question="",model="claude-3-opus-20240229", system="You are an AI assistant with a passion for creative writing and storytelling. Your task is to collaborate with users to create engaging stories, offering imaginative plot twists and dynamic character development. Encourage the user to contribute their ideas and build upon them to create a captivating narrative.",multimodal=None):
    message=claude3_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=max_tokens,
            temperature=1,
            system=system,
            messages=[
                {"role":"user","content":question}    
            ]
)
    return message.content[0].text

def call_openai(max_tokens=4096,model_name="gpt-4-0125-preview",question="",multimodal=None,system="You are an AI assistant with a passion for creative writing and storytelling. Your task is to collaborate with users to create engaging stories, offering imaginative plot twists and dynamic character development. Encourage the user to contribute their ideas and build upon them to create a captivating narrative.",add_image_explain=False):
    if multimodal is None:
        response=openai_client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":question}
        ]
        
        )
    else:
        if add_image_explain:
            question=question+"\n and here is the illustration of the current scene. Use this picture to complement your imagination."

        response=openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        max_tokens=max_tokens,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":[{"type":"text","text" : question},{"type":"image_url","image_url":{"url":multimodal}}] 
                },

        ]

        )

    return response.choices[0].message.content






def init_mistral(load_mistral_model_name="None"):
    
    model = MistralForCausalLM.from_pretrained(load_mistral_model_name,device_map="auto")
    
    return model

def call_mistral(model_name=None, system=None, max_tokens=4096,question="",multimodal=None):
    inp_ids=tokenizer(question,return_tensors="pt")['input_ids']
    len_inp=len(inp_ids[0])
    result=model_name.generate(inp_ids, max_tokens=max_tokens)
    result=tokenizer.decode(result[0,len_inp:],skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return result

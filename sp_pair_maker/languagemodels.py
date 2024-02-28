from openai import OpenAI
import secret
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
import override_bart

class DefaultLM():
    def __init__(self):
        return
    def get_prompt(self,sentence,new_story):
        return sentence
    def update(self,probs):
        return

class OPENAI(DefaultLM):
    def __init__(self,):
        self.api_key=secret.Secret().get_api_key()
        self.client=OpenAI(
         api_key=self.api_key
         )
        self.MODEL = "gpt-3.5-turbo"
        self.system_content="You are a helpful and technical assistant."
        self.user_content="What is the most appropriate prompt for DALL-E-3 to illustrate this sentence?"
        
    def get_propmt(self,sentence,new_story):
        response = self.client.chat.completions.create(
        model=self.MODEL,
        messages=[
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": self.user_content},
        ],
        temperature=0,
        )
        #print(response.choices[0].message.content)
        return response.choices[0].message.content

class MyLanguageModel(DefaultLM):
    def __init__(self,):
        self.lm = override_bart.MyCustomBartConditionalGeneration()
        
        #BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        

    def get_prompt(self,sentence,new_story):
        input_ids=self.tokenizer([sentence],max_length=1024,return_tensors="pt")
        if new_story:
            self.story_memory=None
            self.visual_memory=None
            self.last_sentence=None
            self.last_image=None
        

        prompt=self.lm.generate(input_ids=input_ids["input_ids"], attention_mask=input_ids["attention_mask"],story_memory=self.story_memory, visual_memory=self.visual_memory, last_sentence=self.last_sentence, last_image=self.last_image,fme=self.fme)
        # ... generation은 gradient가 없는데??
        
        print(prompt)
        return prompt

    def update(self,probs):
        # probs 자체가 cos sim이라서 reward이다.
        self.lm.update(reward=probs)

            


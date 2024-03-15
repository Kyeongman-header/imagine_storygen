from .secret import Secret
import torch
from openai import OpenAI
import pickle
import anthropic


claude3_client=anthropic.Anthropic(
        api_key=Secret().get_claude3_api_key()
        )

openai_client=OpenAI(
        api_key=Secret().get_openai_api_key()
        )

def call_claude3(max_tokens=2000,question="",system="You are an AI assistant with a passion for creative writing and storytelling. Your task is to collaborate with users to create engaging stories, offering imaginative plot twists and dynamic character development. Encourage the user to contribute their ideas and build upon them to create a captivating narrative."):
    message=claude3_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=max_tokens,
            temperature=1,
            system=system,
            messages=[
                {"role":"user","content":question}    
            ]
)
    return message.content

def call_openai(max_tokens=2000,model_name="gpt-3.5-turbo",question="",system="You are an AI assistant with a passion for creative writing and storytelling. Your task is to collaborate with users to create engaging stories, offering imaginative plot twists and dynamic character development. Encourage the user to contribute their ideas and build upon them to create a captivating narrative."):
    response=openai_client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[
            {"role":"system","content":system}
            {"role":"user","content":question}
        ]

    )

    return completion.choices[0].message.content




    

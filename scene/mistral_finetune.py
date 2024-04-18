from transformers import AutoTokenizer, MistralForCausalLM, Trainer, TrainingArguments
from trl import SFTTrainer
import json
from datasets import load_dataset
import torch
torch.set_printoptions(profile="full")
import argparse
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer, util
import os
import jsonlines
import pickle
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime
from .secret import Secret

sentence_bert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",device="cuda")
# cos=torch.nn.CosineSimilarity(dim=1,eps=1e06)


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",padding_side='left')
# left side is default(I don't understand why...)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# pad_token_id=tokenizer('[PAD]',return_tensors="pt")['input_ids'][0][1].item()

current_dir =os.path.abspath(__file__)
log_dir = "mistral_logs"

MAX_LENGTH=1500

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, questions, answers, tokenizer, max_length):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        # Tokenize inputs and labels
        inputs = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten()
            "labels" : inputs["input_ids"].copy().flatten(),
        }

def load_data(name,):
    input_texts=[]
    whole_texts=[]
    output_texts=[]
    texts=[]
    length=[]
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset/", name), "r",) as f:
        # taget_json = json.load(f)
        # print(len(f))  # list의 길이 1 출력
        for line in f:
            input_texts.append(line["messages"][0]["content"])
            output_texts.append(line["messages"][1]["content"])
    
    

    dataset=CustomDataset(input_texts,output_texts,tokenizer,max_length=MAX_LENGTH)
    # dataset=load_dataset("json", data_files=os.path.join(os.path.dirname(current_dir), "dataset/", name),)
    # SFTTrainer를 써보려 했는데, 왜인지 안된다.


    return input_texts, output_texts, dataset

def formatting_func(example):
    text = example['messages'][0]['content'] + example['messages'][1]["content"]
    return text

def train(epochs,name,batch_size,model, save_new_model_name):

    # _ , _, batch_whole_ids,batch_label_ids=load_data(name,batch_size)
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    # one_epoch_steps = (len(batch_whole_ids) // batch_size)
    # total_loss=0
    # for epoch in range(epochs):
    #     for i, (inp, outp) in enumerate(zip(batch_whole_ids, batch_label_ids)):
    #         #inp, outp -> (B,longest_whole_length)
    #         #outp는 inp와 동일하되 prompt 부분만 -100으로 마스킹된 버전이다.
    #         inp=inp.to('cuda:0')
    #         outp=outp.to('cuda:0')
    #         # print(inp.shape)
    #         # print(outp.shape)

    #         result=model(inp,labels=outp)
    #         optimizer.zero_grad()
    #         result.loss.backward()
    #         optimizer.step()

    #         print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {result.loss.item()}")
    #         total_loss+=result.loss.item()
            
    #         if (i+1) % 100==0:
    #             total_loss=0
    #             writer.add_scalar("Loss/train", total_loss/100, epoch * one_epoch_steps + i)

    #         inp=inp.to('cpu')
    #         outp=outp.to('cpu')

    #         # del inp
    #         # del outp
    #         # torch.cuda.empty_cache()
    #     # torch.save({'model_state_dict' : model.state_dict()}, os.path.join(os.path.dirname(current_dir), "models", save_new_model_name),)
    #     model.push_to_hub(save_new_model_name)
    #     print("\nsave done!")

    _,_,train_dataset=load_data(name,)

    training_args = TrainingArguments(
    push_to_hub=True,
    output_dir=save_new_model_name,
    hub_token=Secret().get_huggingface_token_key(),
    hub_strategy="end",
    save_strategy ="epoch",

    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_dir=os.path.join(os.path.dirname(current_dir), log_dir),
    logging_steps=100,
    save_total_limit=2,
    overwrite_output_dir=True,
    )
    trainer = Trainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    # max_seq_length=MAX_LENGTH,
    )

    trainer.train()

    print("\ntrain done!")
    model.push_to_hub(save_new_model_name)
    print("\nsave done!")



def eval(name,batch_size,model):
    input_texts, output_texts,_ =load_data(name,batch_size)
    avg_cos=0
    for i, (inp, outp) in enumerate(zip(input_texts,output_texts)):
            inp_ids=tokenizer(inp,return_tensors="pt")['input_ids']
            len_inp=len(inp_ids[0])
            
            result=model.generate(inp_ids, max_new_tokens=4000)
            
            result=tokenizer.decode(result[0,len_inp:],skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print("generation result : ")
            print(result)
            result_embedding = sentence_bert.encode(result)
            outp_embedding = sentence_bert.encode(outp)
            print(result_embedding.shape)
            print(outp_embedding.shape)
            c=util.cos_sim(result_embedding,outp_embedding).item()
            avg_cos += c
            print(c)

            r=[{'query' : inp, 'generation_result' : result, 'golden_result' : outp}]

            if os.path.exists(current_dir+"/results/"+ "mistral_eval_result.jsonl"):
                mode="a"
            else:
                mode="w"
            with jsonliens.open(current_dir + "/results/"+ "mistral_eval_result.pickle",mode) as fw:
                fw.write_all(r)

            # input()
    print("avg_cos")
    print(avg_cos)
    writer.add_scalar("Loss/train", loss.item(), epoch * one_epoch_steps + i)


# Usage Instruction.

# dataset => dataset name, such as "writingPrompts", "reedsyPrompts", "booksum", "writingPrompts_reedsyPrompts_booksum".

# batch_size => you may have to designate this size not beyond 2, because the model size and each batch size is very big.

# epochs => 3 epochs use to take 3~4 days for training.

# save_new_model_name => the name of huggingface's hub repo to push finetuned mistral. if you don't specify it, it is automatically set as today date.

# load_past_model_name => if you want continual learning or just want to evaluate, 
# designate name such as "Iyan/2024-04-20"(don't forget to add 'user name/'!).
# if you don't set this, the base pretrained mistralai's model is loaded.
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="writingPrompts", action="store")
    parser.add_argument("-batch_size", "--batch_size", default=16,type=int, action="store")
    parser.add_argument("-epochs", "--epochs", default=3,type=int, action="store")
    parser.add_argument("-save_new_model_name", "--save_new_model_name", default=datetime.now().date().strftime("%Y-%m-%d"), action="store")
    parser.add_argument("-load_past_model_name", "--load_past_model_name", default="mistralai/Mistral-7B-v0.1", action="store")
    parser.add_argument("-eval_only", "--eval_only", default=False, action="store_true")
    
    args = parser.parse_args()
    
    # model="None"
    model = MistralForCausalLM.from_pretrained(args.load_past_model_name,device_map="auto")
    # model.resize_token_embeddings(len(tokenizer))

    if args.eval_only != True:
        train(args.epochs,args.dataset + "_train.jsonl",args.batch_size,model,args.save_new_model_name)
    
    eval(args.dataset + "_valid.jsonl",args.batch_size, model,)
    writer.close()
    print("train done.")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
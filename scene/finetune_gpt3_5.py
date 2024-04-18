from openai import OpenAI
import os
from .call import *
import argparse
import jsonlines
from datetime import datetime
current_dir = os.path.abspath(__file__)



def file_maker(name):
    print(os.path.join(os.path.dirname(current_dir), "dataset/"+name))
    file_information=openai_client.files.create(
    file=open(os.path.join(os.path.dirname(current_dir), "dataset/"+name ), "rb"),
    purpose="fine-tune"
    )
    print(file_information)
    return file_information.id

def finetune(my_model_name,train_file,valid_file):
    finetune_information=openai_client.fine_tuning.jobs.create(
        training_file=train_file,
        model="gpt-3.5-turbo",
        suffix=my_model_name,
        validation_file=valid_file,
        )
    print(finetune_information)
    return finetune_information





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_make", "--file_make", default=False, action="store_true")
    parser.add_argument("-dataset", "--dataset", default="writingPrompts", action="store")
    parser.add_argument("-suffix", "--suffix", default=datetime.now().date().strftime("%Y-%m-%d"), action="store")
    args = parser.parse_args()

    if args.file_make : 
        train_id=file_maker(args.dataset + '_train.jsonl')
        valid_id=file_maker(args.dataset + '_valid.jsonl')
        test_id=file_maker(args.dataset + '_test.jsonl')
        lines=[]
        name=args.dataset + "_train"
        lines.append({name : train_id})
        name=args.dataset + "_valid"
        lines.append({name : valid_id})
        name=args.dataset + "_test"
        lines.append({name : test_id})
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), "file_id.jsonl"), mode='a') as f:
            f.write_all(lines)


    else:
        ids=[]
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), "file_id.jsonl"), "r",) as f:
            for line in f:
                ids.append(line)
        
        for id in ids:
            if args.dataset + "_train" == list(id.keys())[0]:
                train_id=id[args.dataset+"_train"]
            elif args.dataset + "_valid" == list(id.keys())[0]:
                valid_id=id[args.dataset+"_valid"]
            elif args.dataset + "_test" == list(id.keys())[0]:
                test_id=id[args.dataset+"_test"]

        print(train_id)
        information=finetune(args.suffix, train_id, valid_id)
        ftid={args.suffix : information.id}
        lines=[ftid]
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), "finetune_job_id.jsonl"), mode='a') as f:
            f.write_all(lines)

    # print(openai_client.fine_tuning.jobs.list())
    

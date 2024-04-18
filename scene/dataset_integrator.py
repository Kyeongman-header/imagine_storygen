import os
import argparse
import jsonlines

current_dir =os.path.abspath(__file__)

def integrator(dataset_list):
    typ=["train","valid","test"]
    name=('_').join(dataset_list)

    for t in typ:
        _lines=[]
        _point_lines=[]
        for dataset in dataset_list:
            with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset/", dataset+"_"+ t +".jsonl"), "r",) as f:
            # taget_json = json.load(f)
            # print(len(f))  # list의 길이 1 출력
                for line in f:
                    _lines.append(line)

            with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset/", dataset+"_"+ t +"_points.jsonl"), "r",) as f:
            # taget_json = json.load(f)
            # print(len(f))  # list의 길이 1 출력
                for line in f:
                    _point_lines.append(line)
    
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name + "_" + t + ".jsonl"), mode='w') as f:
            f.write_all(_lines)
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name + "_" + t + "_points.jsonl"), mode='w') as f:
            f.write_all(_point_lines)
        

import ast
def arg_as_list(s):
    v=ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list." % (s))
    return v

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_list", "--dataset_list", default=["writingPrompts","reedsyPrompts","booksum"], type=arg_as_list,)
    
    args = parser.parse_args()
    integrator(args.dataset_list)
    print("successfully done.")
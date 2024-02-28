import os
import nltk
nltk.download('punkt')  # You only need to download the punkt tokenizer once
from nltk.tokenize import sent_tokenize
import pickle
from tqdm import tqdm, trange

def load_and_sentence_split():

    data = ["test","train", "valid"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "writingPrompts")
    for name in data:
        
        sentence_wp=[]
        print(name + " dataset.")
        print("load start.")
        with open(folder_path+'/'+name + ".wp_target") as f:
            stories = f.readlines()
        with open(folder_path+'/'+name + ".wp_source") as f:
            titles = f.readlines()
        print("load end.")
        print("sent_tokenizing start.")
        for story,title in tqdm(zip(stories,titles)):
            sentences=sent_tokenize(story)
            sentence_wp.append({"title" : title, "sent_prompt_pair" : [{"sentence" : sent,"prompt" :""} for sent in sentences]})
        
        print("sent_tokenizing end.")

        print("start saving.")
        # save as 'dataset/sentence_WP/train.pickle'
        f = open(current_dir + '/' + "sentence_WP/"+name+'.pickle',"wb")
        pickle.dump(sentence_wp,f)
        f.close()
        print("end saving.")


if __name__ == "__main__":
    load_and_sentence_split()

import jsonlines
import json
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import argparse
from .call import *
from tqdm import tqdm, trange

current_dir = (os.path.abspath(__file__))

def summary_maker(name,):
    _lines=[]
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset",name)) as f:
        for line in f:
            _lines.append(line)

    for j,l in enumerate(tqdm(_lines)):
        summaries=[]
        for s in l['stories']:
            question = "Summarize this text in 1-2 sentences. Note that this is a part of one story. DO NOT MAKE any explanation or other words. just give me the summary."
            question+= "\nText : "
            question+= s
            # print(question)
            try:
                summary=call_openai(question=question,model_name="gpt-3.5-turbo")
            except Exception as e:
                print(e)
                print("pass this example.")
                summaries.append("(Not given.)")
                continue
                # print("call claude 3")
                # summary=call_claude3(question=question,model_name="claude-3-sonnet-20240229")
            # print("Answer")
            # print(summary)
            # input()
            
            summaries.append(summary)
            
        _lines[j]['summaries']=summaries
    
    print(len(_lines))
    
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name), mode='w') as f:
        f.write_all(_lines)

def extracter(name,):
    input_texts=[]
    _lines=[]
    error_flag=False
    count=0
    
            
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name)) as f:
        for line in f:
            # print(line)
            count+=1
            
            if list(line.keys())[0]=='message':
                error_flag=True
                line['messages']=line['message']
                del line['message']
            
            _lines.append(line)

    if error_flag:
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name), mode='w') as f:
            f.write_all(_lines)
        print("message's' error fix done.")

    error_flag=False
    count=0
    _lines=[]
    first=""
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name)) as f:
        for line in f:
            # print(line)
            
            count+=1
            if first==line['messages'][0]['content']:
                error_flag=True
                break
            if count==1:
                first=line['messages'][0]['content']
            _lines.append(line)

    if error_flag:
        print(len(_lines))
        print("length truncate")
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name), mode='w') as f:
            f.write_all(_lines)
    
    
    error_flag=False
    _lines=[]
    count=0
    flag=False
    story_begin=0
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name)) as f:
        before_line=[]
        for line in f:
            # print(line)
            _lines.append(line)
            if before_line!=[]:
                if ("going to make ending of a story" in line['messages'][0]['content'] or "going to make body of a story" in line['messages'][0]['content']) and flag is False:
                    flag=True
                    story_begin=count
                
                if ("going to make ending of a story" in before_line['messages'][0]['content'] or "going to make body of a story" in before_line['messages'][0]['content']) and 'A story has some interesting characters' in line['messages'][0]['content']:
                    
                    for j in range(story_begin,count-1,):                  
                        _lines[j]['messages'][0]['content']=_lines[j]['messages'][0]['content'].replace("going to make ending of a story.", "going to make body of a story.")
                    _lines[count-1]['messages'][0]['content']=_lines[count-1]['messages'][0]['content'].replace("going to make body of a story.", "going to make ending of a story.")
                    
                    flag=False
                    story_begin=0
            count+=1
            
            before_line=line

    
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name), mode='w') as f:
        f.write_all(_lines)
    print("write done.")


    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name)) as f:
        for line in f:
            input_texts.append(line["messages"][0]["content"] + "ANSWER. " + line["messages"][1]["content"]) 
    


    documents_informations=[]
    
    one_doc_inf={}
    stories=[]
    whole_error_count=0
    error_flag=False
    for input_text in input_texts:
        
        if "A story has some interesting characters" in input_text:
            
            if len(stories) !=0 :
                print(len(stories))
                one_doc_inf['stories']=stories
                if len(one_doc_inf['stories'])!= int(one_doc_inf['length']):
                    one_doc_inf['length']=str(len(one_doc_inf['stories']))
                documents_informations.append(one_doc_inf)
                # print(documents_informations[-1])
                # input()
            error_flag=False
            stories=[]
            one_doc_inf={}
            sents=sent_tokenize(input_text)
            sents=[sent.split('\n') for sent in sents]
            
            sents=[x for xs in sents for x in xs]
            for i,sent in enumerate(sents):
                # print(sent)
                # input()
                if "the total number of scenes in the story" in sent:
                    words=word_tokenize(sent)
                    flag=True
                    for w in words:
                        if w.isdigit() and flag:
                            length=w
                            flag=False
                    emotions=sent.replace("story :","story").replace("the total number of scenes in the story","").replace("emotions that will appear in our story","").replace("1.","").replace("2.","").replace(length,"").replace("\n","").replace("    Emotions","Emotions")
                    
                    
                elif "backgrounds that will appear in our story" in sent:
                    backgrounds=sent.replace("story :","story").replace(" backgrounds that will appear in our story","").replace("3.","")
                    
                    
                elif "senses that will appear in our story" in sent:
                    senses=sent.replace("story :","story").replace(" senses that will appear in our story","").replace("4.","")
                    
            
            # print(length)
            # print(emotions)
            # print(senses)
            # print(backgrounds)
            
            
            one_doc_inf['length']=length
            one_doc_inf['emotions']=emotions
            one_doc_inf['backgrounds']=backgrounds
            one_doc_inf['senses']=senses
            # input()

        elif "We are goint to make a central plot of our story." in input_text:
            sents=sent_tokenize(input_text)
            for i,sent in enumerate(sents):
                if "characters that will apear in our story" in sent:
                     characters=sent.replace("story :","story").replace("characters that will apear in our story","").replace("5.","").replace("\n","")

            # print(characters)
            one_doc_inf['characters']=characters
            # input()
        elif "Now, let's create the scenes to structure the story." in input_text:
            sents=sent_tokenize(input_text)
            for i,sent in enumerate(sents):
                if "the summary or overview of our story" in sent:
                     plot=sent.replace("story :","story").replace("the summary or overview of our story","").replace("6.","").replace("\n","")
            # print(plot)
            one_doc_inf['plot']=plot
            # input()
        elif "For each scene in a story, the main characters take several actions for his or her purpose, or for survival, or just because of their habits" in input_text:
            sents=sent_tokenize(input_text)
            start=False
            event_start=False
            scene=[]
            scenes=[]
            event=[]
            events=[]
            answer_flag=False
            for i,sent in enumerate(sents):
                # if "Scene " in sent and ": Emotions :" in sent:
                if "ANSWER." in sent:
                    answer_flag=True
                if answer_flag != True:
                    if "Scene" in word_tokenize(sent)[0] or "Breif summaries of our scenes" in sent:
                        # print(sent)
                        scene.append(sent.replace('Breif summaries of our scenes :',""))
                        start=True
                    elif "_END." in sent and start:
                        # print(sent)
                        scene.append(sent)
                        scenes.append((' ').join(scene))
                        start=False
                        scene=[]
                    elif start :
                        scene.append(sent)
                

                if answer_flag:
                    if "Event" in word_tokenize(sent)[0] or "Events : " in word_tokenize(sent)[0]:
                        event_start=True
                        event.append(sent)
                    elif "_END." in sent and event_start:
                        event.append(sent)
                        events.append((' ').join(event))
                        event_start=False
                        event=[]
                    elif event_start:
                        event.append(sent)



            # print(len(scenes))
            # print(len(events))

            if len(scenes)!=len(events) or len(scenes)!=int(one_doc_inf['length']) or len(events)!=int(one_doc_inf['length']):
                whole_error_count+=1
                error_flag=True
                continue

            one_doc_inf['scenes']=scenes
            one_doc_inf['events']=events
            
            # input()
        elif "These are informations about current scene you need to write" in input_text and error_flag is False:
            sents=sent_tokenize(input_text)
            start=False
            story=[]
            for i,sent in enumerate(sents):
                if start:
                    story.append(sent)
                if "ANSWER" in sent:
                    start=True

            story=(' ').join(story)
            ws=word_tokenize(story)
            splited_stories=[]
            if len(ws)>2500:
                while len(ws)>2500:
                    splited_stories.append((' ').join(ws[:2500]))
                    ws=ws[2500:]
            
                splited_stories.append((' ').join(ws[:2500]))
            


            if len(splited_stories)>0:
                target=len(stories)
                
                cp_scenes=one_doc_inf['scenes'][target]
                cp_events=one_doc_inf['events'][target]

                
                # print(len(one_doc_inf['scenes']))
                # print(one_doc_inf['scenes'])
                scenes=one_doc_inf['scenes'][:target] + [cp_scenes] * len(splited_stories) + one_doc_inf['scenes'][target+1:]
                events=one_doc_inf['events'][:target] + [cp_events] * len(splited_stories) + one_doc_inf['events'][target+1:]
                
                
                one_doc_inf['scenes']=scenes
                one_doc_inf['events']=events
                
                # print(len(one_doc_inf['events']))
                # print(one_doc_inf['scenes'])

            if len(splited_stories)>0:
                stories.extend(splited_stories)
            else:
                # print("무사 통과")
                stories.append(story)
    
    print("total valid dataset number " + str(len(documents_informations)))
    print("error number " + str(whole_error_count))
    # input()
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", name.replace('.jsonl','_points.jsonl')), mode='w') as f:
        f.write_all(documents_informations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="writingPrompts", action="store")
    parser.add_argument("-summarize", "--summarize", default=False, action="store_true")
    args = parser.parse_args()

    extracter(args.dataset + "_valid.jsonl")
    extracter(args.dataset + "_test.jsonl")
    extracter(args.dataset + "_train.jsonl")

    if args.summarize:
        summary_maker(args.dataset + "_valid_points.jsonl")    
        summary_maker(args.dataset + "_test_points.jsonl")
        summary_maker(args.dataset + "_train_points.jsonl")
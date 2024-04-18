import pickle
import argparse
import os
from tqdm import tqdm, trange
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import json
from collections import OrderedDict
import math
from .call import *
import unicodedata


def maker(dataset="writingPrompt", model="claude3", few_number=100):
    
    total_cost=0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if "claude" in model:
        caller=call_claude3
    else:
        caller=call_openai

    if "writingPrompt" in dataset or "writing" in dataset or "Writing" in dataset:
        train_data=os.path.join(os.path.dirname(current_dir), "dataset/writingPrompts", "train.wp_target")
        valid_data=os.path.join(os.path.dirname(current_dir), "dataset/writingPrompts","valid.wp_target")
        test_data=os.path.join(os.path.dirname(current_dir), "dataset/writingPrompts","test.wp_target")
        #data=[train_data,valid_data,test_data]
        data=[valid_data,test_data,train_data]
        for k,name in enumerate(data):
            with open(name) as f:
                stories = f.readlines()

                if k!=2:
                    stories = stories[:few_number // 10]
                else:
                    stories = stories[:few_number]
                print(len(stories))


            for story in tqdm(stories):
                if k==0:
                    total_cost+=jsonl_maker(story,'writingPrompts_valid.jsonl',current_dir,caller=caller)
                elif k==1:
                    total_cost+=jsonl_maker(story,'writingPrompts_test.jsonl',current_dir,caller=caller)
                else:
                    total_cost+=jsonl_maker(story,'writingPrompts_train.jsonl',current_dir,caller=caller)

    elif "reedsy" in dataset or "Reedsy" in dataset:
        train_data=os.path.join(os.path.dirname(current_dir), "dataset/reedsyPrompts", "reedsy_prompts_whole.pickle")
        with open(train_data,"rb") as fi:
            reedsy = pickle.load(fi)

        
        reedsy_len=len(reedsy)
        
        reedsy_len=few_number
        test_len=math.ceil(reedsy_len*0.1)
        valid_len=test_len*2

        
        reedsy_len+=valid_len

        print(reedsy_len)
        input()

        for k,line in enumerate(reedsy):
            story=unicodedata.normalize("NFKD", line['story'])
            if k<reedsy_len-valid_len: 
                total_cost+=jsonl_maker(story,"reedsyPrompts_train.jsonl",current_dir,caller=caller)
            elif k<reedsy_len-test_len:
                total_cost+=jsonl_maker(story,"reedsyPrompts_valid.jsonl",current_dir,caller=caller)
            else :
                total_cost+=jsonl_maker(story,"reedsyPrompts_test.jsonl",current_dir,caller=caller)

            if k==reedsy_len:
                break

    elif "booksum" in dataset or "Booksum" in dataset or "Book" in dataset or "book" in dataset :
        train_data=os.path.join(os.path.dirname(current_dir), "dataset/booksum")



        for (path, d, files) in os.walk(train_data):
            booksum_len=len(d)
            booksum_len = few_number
            
            
            test_len=math.ceil(booksum_len*0.1)
            valid_len=test_len*2
            
            booksum_len+=valid_len

            print(booksum_len)
            input()

            for k,b in tqdm(enumerate(d)):
                story_file=path+'/'+b + '/book_clean.txt'
                story=""
                with open(story_file,"r") as f:
                    story=f.readlines()
                story=' '.join(story)

                if k<booksum_len - valid_len: 
                    total_cost+=jsonl_maker(story,"booksum_train.jsonl",current_dir,caller=caller)
                elif k<booksum_len-test_len:
                    total_cost+=jsonl_maker(story,"booksum_valid.jsonl",current_dir,caller=caller)
                else:
                    total_cost+=jsonl_maker(story,"booksum_test.jsonl",current_dir,caller=caller)

                if k == booksum_len:
                    break

    print("total cost : ")
    print(str(total_cost)  + "$")
        

def jsonl_maker(story,jsonl_name,current_dir,caller=call_claude3,):
    
    from_Plot_points_to_Characters=[]
    from_Plot_points_to_Main_plot=[]
    from_Plot_points_to_Scene_informations=[]
    from_Scene_informations_to_Events=[]
    from_Scene_information_to_Scene=[]
    total_cost=0
    if caller==call_claude3:
        in_price=15
        out_price=75
        if len(word_tokenize(story)) > 197000: # 최대 200k의 입력 컨텍스트.
            story=(' ').join(word_tokenize(story)[:197000])
            print("this story can exceed the limit of length of api.")
    elif caller==call_openai:
        in_price=10
        out_price=30

        if len(word_tokenize(story)) > 14000: # 최대 16.385k의 입력 컨텍스트.
            story=(' ').join(word_tokenize(story)[:14000])
            print("this story can exceed the limit of length of api.")
    else:
        in_price=0
        out_price=0

    
    sentences=sent_tokenize(story)
    sentence_label_story=(' ').join(['[' + str(i) + '] ' + sentences[i]  for i in range(len(sentences))])
    
    # print("sentence_label_story_len : ")
    # # print(sentence_label_story)
    # # print("story : ")
    # # print(story)

    # print(str(len(word_tokenize(story))/1000000) + "M")
    

    question="There are dominant emotions, senses, backgrounds in a story. for example, a scene has 'sadness' emotion, 'cold and dry' sense, and 'city' background. also, a story has different interesting characters. Each characters has different personality, appearance, habits, etc. for example, 'Alex : male, 26 years old, cynical, tall and handsome, a heavy smoker, a hunter, lost his daughter when he was away, raised without parents and experienced war, left-handed, ...'. Lastly, a story has a main plot, which can be seen as total summary of it."
    question+="\nHere are the questions. Answer these 5 questions as succinctly and clearly as possible, without unnecessary elaboration. Example format of your answer : 'Emotions : sadness, emptiness. Senses : cold. Backgrounds : city. Characters : Alex : male, 26 years old, cynical, tall and handsome, a heavy smoker, a hunter. Plot : A story of a man who finally becomes god.' Don't forget to start each answer with the topic words, 'Emotions', 'Senses','Backgrounds','Characters',and 'Plot'. Keep the format of answer. you don't have to repeat or to make a summary of your answer."
    question+="\n1. What emotions are revealed in this story? Remember that in longer story, there can also be various opposing emotions coexisting."
    question+="\n2. What senses are revealed in this story? Keep in mind that in longer story, various opposing senses can coexist."
    question+="\n3. What backgrounds are revealed in this story? Keep in mind that in longer story, various backgrounds can exist."
    question+="\n4. Who are the characters appearing in this story? List all possible main characters and describe their attributes (e.g., personality, appearance, background, traits, habits, etc.) in as much detail as possible, without inventing anything not found in the text."
    question+="\n5. How would you summarize the entire story in one or two sentences? For instance, how could you introduce this story in just a couple of sentences?"
    question+="\nAnd here's the story where you need to find the answers to the questions given below. \n"
    # print("Question : ")
    # print(question)
    
    plot_points=caller(max_tokens=4096,question=question + story)
    # plot_points에서 한꺼번에 senses, emotions, backgrounds, characters, main plot까지 뽑아낸다.
    # print("Answer : ")
    # print(plot_points)

    cost=len(word_tokenize(question))/1000000
    total_cost+=cost*in_price
    cost=len(word_tokenize(plot_points))/1000000
    total_cost+=cost*out_price
    
    # print(str(total_cost)  + "$")

    
    words=word_tokenize(plot_points)
    pp=[""]
    
    characters=""
    plot=""
    character_flag=False
    plot_flag=False
    for w in words:
        
        if w=='Senses' or w=='Backgrounds':
            pp.append("")
        if w=='Characters' or w=='Character':
            character_flag=True
        if w=='Plot' :
            character_flag=False
            plot_flag=True

        if character_flag:
            characters+=w+" "
        elif plot_flag:
            plot+=w+" "
        else:
            pp[-1]+=w+" "
    
    dic_plot_points={}
    dic_plot_points['emotions']=pp[0]
    dic_plot_points['senses']=pp[1]
    dic_plot_points['backgrounds']=pp[2]
    
    plot_points=(' ').join(pp)

    # print(plot_points)
    # print(dic_plot_points)
    # print(characters)
    # print(plot)
    #input()
    
    question="A story is composed of many scenes. Short stories usually have only 2-5 scenes, but long stories typically have 50-100 or even more scenes. The story below is labeled with the number of each sentence, such as [3], [15], at the beginning of each sentence. When you think about it, answer with the sentence label where the scene changes, and using those divisions, state how many scenes you believe the story is composed of."
    question+="\nYou must set the scenes to be composed of at least 2 sentences and at most 15 sentences. Answer the questions as succinctly and clearly as possible, without unnecessary elaboration. Remember, all you need to provide are the label numbers of the sentences where the scene changes and the number of scenes. So, refrain from adding any unnecessary words or phrases. Here's an format of your response: 'Scene labels : ... Number of scenes : .... '. You must always include the label of the last sentence in the Scene labels."
    question+="\nAnd here's the story where you need to find the answers to the questions given below.\n"
    
    # print("Question : ")
    # print(question)

    sentences_numbers=caller(max_tokens=4096,question=question + sentence_label_story)
    # 여기서 뽑아낸 sentences_number로 scene들을 나눈다. 여기서 scene이 나온다.
    # print("Answer : ")
    # print(sentences_numbers)
    
    cost=len(word_tokenize(question))/1000000
    total_cost+=cost*15
    cost=len(word_tokenize(sentences_numbers))/1000000
    total_cost+=cost*75
    
    # print(str(total_cost)  + "$")

    words=word_tokenize(sentences_numbers)
    num_of_scenes=""
    scene_labels=[]
    numofscene_flag=False
    
    for w in words:

        if w=='Number':
            numofscene_flag=True
            
        if numofscene_flag:
            if w.isdigit():
                num_of_scenes=int(w)
        else:
            if w.isdigit():
                scene_labels.append(int(w))
    
    count=0
    s=""
    scenes=[]


    # api가 잘못 답을 한 경우로, scene_labels의 마지막이 실제 마지막 문장이 아닌 경우.
    if scene_labels[-1]!=len(sentences)-1:
        scene_labels.append(len(sentences)-1)


    # api가 잘못 답을 한 경우로, num_of_scenes가 틀린 경우.
    if num_of_scenes !=len(scene_labels):
        num_of_scenes=len(scene_labels)


    for j in range(len(sentences)):
        
        s+=sentences[j]+" "
        if j==scene_labels[count]:
            scenes.append(s)
            s=""
            count+=1
    
    # print(scenes)
    # print(len(scenes))
    # print(num_of_scenes)
    #input()
    scenes_informations=[]
    events=[]
    explains=[]

    for j,scene in enumerate(scenes):
        
        
        question="Examine a scene from the entire story and provide the emotions, senses, background, names of characters, and key events or plot points that can be felt or identified in that scene."
        question+="\nFurthermore, provide an explanation for why emotions, sensations, backgrounds, and characters have changed compared to previous scenes. If it's the first scene, analyze why emotions, sensations, backgrounds, and characters appeared in that scene. Also, similar to the previous one, analyze why the key events occurred considering their relationship with the preceding scenes. All explanations must start with the word 'Because.' For example, if Alex appeared in the previous scene but not in this one, you should explain the reason starting with a sentence beginning with 'Because'."
        if j!=0:
            question+="\nAdditionally, you will be provided with a list of emotions, senses, backgrounds, characters, and key events or plot points found in the immediately preceding scene. If anything in the current scene is unclear, carefully assess whether the emotions, senses, backgrounds, and characters from the previous scene continue. If there is sufficient evidence that they do, answer based on the elements from the previous scene."
        question+="\nAnd here's the scene where you need to find the answers to the questions given below.\n"
        question+=scene
        if j!=0:
            question+="\nAnd these are the emotions, senses, background, names of characters, and key events or plot points in the immediately preceding scene."
            question+=scenes_informations[-1]
            question+="\nAnd here's the full text of immediately preceding scene.\nfull text : "
            question+=scenes[j-1]
        question+="\nNow, Illustrate emotions, senses, background, names of characters, major events, or plot succinctly and clearly in sequential order. And also provide explanations for the differences between the scenes mentioned earlier. For the first Explain, analyze why emotions, senses, background, and characters have changed using sentences starting with Because. For the second Explain, analyze why the key event occurred considering its relationship with the preceding scenes in the same manner. Here's an example of your response:  'Emotions : sadness, emptiness. Senses : cold. Backgrounds : city. Characters : I(me). Event : walking in the city. Explain of Scene: Because he was moving from the town into the city, the background is changed. Explain of Event: Because he has been doing so since the previous one, he is still wandering aimlessly in this scene. ' Don't forget to end each sentence with full stop, '.'. Keep the format of the example. \n"
        
        # print("Question : ")
        # print(question)

        scene_information=caller(max_tokens=4096,question=question)
        # scenes_information에서 한꺼번에 events도 뽑아낸다.
        # print("Answer : ")
        # print(scene_information)
        # 이후 scenes_information에서 event는 분리해 낸다.
        cost=len(word_tokenize(question))/1000000
        total_cost+=cost*15
        cost=len(word_tokenize(scene_information))/1000000
        total_cost+=cost*75
        
        # print(str(total_cost)  + "$")

        information=""
        s=""
        event=""
        scene_explain=""
        event_explain=""
        
        words=word_tokenize(scene_information) 

        for w in words:
            #print(s)
            ##input()

            if (w=='Event' or w=='Events') and len(information)==0:
                information=s
                s=w + " "
            elif w=='Explain' or w=='Explains': 
                if len(event)==0:
                    event=s
                else:
                    scene_explain=s
                s=w + " "
            else:

                s+=w+" "
        event_explain=s
        
        # print("information")
        # print(information)
        # print("event")
        # print(event)
        # print("scene_explain")
        # print(scene_explain)
        # print("event_explain")
        # print(event_explain)


        information+=scene_explain + ' _END.'
        event+=event_explain + ' _END.'

        scenes_informations.append('Scene ' + str(j+1) + " : " + information)
        events.append(event)
        #explains.append(explain)
        # print(scenes_informations)
        # print(events)
        # print(len(scenes_informations))
        
        #input()
        finetune_question="These are informations about current scene you need to write."
        finetune_question+="\n1. short summary or overview of our story : " + plot
        finetune_question+="\n2. characters that will appear in our whole story :  " + characters
        finetune_question+="\n3. Scene you have to implement : " + information + " " + event

        if j == 0:
            
            finetune_question+="\nWe’re going to make a first scene of a story. There are several information of the current scene. With these things, you should make a corresponding sentences that reflect the information of the scene and main events. The sentences’ length should be at least 3 and at most 30 sentences. Keep in mind that this is the introduction of the story. It should contain some attractive and mysterious points to attract readers."

        elif j==len(scenes)-1:
            finetune_question+="We’re going to make body of a story. This is the " + str(j+1) +"-th scene of the total " + str(num_of_scenes) + "scenes in our story. As you can see, there are several information of the current and past scenes. With these things, you should make a corresponding sentences that reflect the information of the scene and main events. The sentences’ length should be at least 5 and at most 30 sentences. Keep in mind that this is the " + str(j+1)+" of " + str(num_of_scenes) + " scenes of the story, it should have progressive, and reasonable sentences that is coherent with the context, but also should be interesting enough that the readers are not sick of our story."
            finetune_question+="\nand the last generated scene is given by default."
            finetune_question+="\nScene #" + str(j) + ". " + scenes[j-1]
        else:
            finetune_question+="\nWe’re going to make ending of a story. This is the " + str(j+1) +"-th scene of the total " + str(num_of_scenes)+ "scenes in our story. \nAs you can see, there are several information of the current and past scenes. With these things, you should make a corresponding sentences that reflect the information of the scene and main events. The sentences’ length should be at least 5 and at most 30 sentences. Keep in mind that this is the end of the story. It should give fascinating climax, or surprising reversal, or appropriate denouement."
            finetune_question+="\nand the last generated scene is given by default."
            finetune_question+="\nScene #" + str(j) + ". " + scenes[j-1]


        # print("\nfor finetune, Question : ")
        # print(finetune_question)
        # print("\nAnswer : ")
        # print(scene)
        from_Scene_information_to_Scene.append([{"role":"user", "content": finetune_question},{"role" : "assistant","content": scene}])
        #input()
    
    



    # Events generation from scene informations.
    
    finetune_question="These are the given scenes of our story, main characters, and short summary or whole plot of our whole story."
    finetune_question+="\n1. Breif summaries of our scenes : " + (' ').join(scenes_informations)
    finetune_question+="\n2. characters that will appear in our story : " + characters
    finetune_question+="\n3. short summary or over view of our total story : " +plot
    finetune_question+="\nFor each scene in a story, the main characters take several actions for his or her purpose, or for survival, or just because of their habits. Also, there may be some expected or unexpected circumstances which affect and change the characters’ attitude, thoughts, and actions. Those main characters’ actions and expected or unexpected circumstances can be called ‘event’, together. Imagine events that can occur in the i-th scene by considering the given information. Each events should be interesting, unexpected, and impressive. Remember, these events must ultimately lead towards a given plot or short summary of our story."
    finetune_question+=" by end of each scene, you should add a reason why each events you wrote is connected to the past scene and is part of the whole plot, starting sentences with the word 'Because'. Each scene you add to end '_END'."
    finetune_question+="\nAnswer this way : SCENE 1. The man is walking through the ruins, but nobody exists there. Explain : Because this is the first scene of our story, this event can be the attractive introduction. _END.  SCENE 2. Suddenly a mysterious girl is watching his back, and the darkness falls. Explain : Because at the last scene the man was walking through the ruins, appearance of a mysterious girl is suitable for an intersting story and the whole plot. _END. SCENE 3. ..."


    # print("\nfor finetune, Question : ")
    # print(finetune_question)
    # print("\nAnswer : ")
    # print((' ').join(events))
    from_Scene_informations_to_Events.append([{"role":"user", "content": finetune_question},{"role" : "assistant","content":(' ').join(events)}])
    #input()


    # Character generation.
    
    finetune_question="A story has some interesting characters. Each character has different characteristics, personality, age, appearance, habits, trauma, growth background, etc. Let’s imagine some characters that will be play roles in our story. We don’t have any specific plot yet, but do have the plot points that will develop into a story. You can freely imagine some characters who are most suitable for our story. Don't limit your imagination to this plot point, but imagine different and diverse characters that might appear in an interesting story. The plot points are provided below. "
    finetune_question+="\n1. the total number of scenes in the story : " + str(num_of_scenes)
    finetune_question+="\n2. emotions that will appear in our story : " + dic_plot_points['emotions']
    finetune_question+="\n3. backgrounds that will appear in our story : " + dic_plot_points['backgrounds']
    finetune_question+="\n4. senses that will appear in our story : " + dic_plot_points['senses']
    finetune_question+="\nTaking into account the length of the story, List some characters you imagined one by one with his or her detailed information. Please remember, don't make too long list. You don't need to make the list as long as the number of scenes. usually there are not more than 2-3 characters in short stories, so the length of your list as well. But if the story is long, there can be more than 3 characters. Long stories means that it has least 50+ scenes. if we have smaller length of scenes than 50, you should consider our story as a short story. and remember, these characters will appear in one story! In a short story, characters that are too disconnected from each other find it challenging to coexist. your answer example: Alex : male, 26 years old, cynical, tall and handsome, a heavy smoker, a hunter, lost his daughter when he was away, raised without parents and experienced war, left-handed, etc.(Only do I need is the list of characters. don't say other things like 'Certainly! Here is a list of possible...' or 'I hope this hepls!')"


    # print("\nfor finetune, Question : ")
    # print(finetune_question)
    # print("\nAnswer : ")
    # print(characters)
    from_Plot_points_to_Characters.append([{"role":"user", "content":finetune_question},{"role" : "assistant","content":characters}])
    #input()

    ### Main Plot Question Answer.
    
    finetune_question="These are the given plot points."
    finetune_question+="\n1. the total number of scenes in the story : " + str(num_of_scenes)
    finetune_question+="\n2. emotions that will appear in our story : " + dic_plot_points['emotions']
    finetune_question+="\n3. backgrounds that will appear in our story : " + dic_plot_points['backgrounds']
    finetune_question+="\n4. senses that will appear in our story : " + dic_plot_points['senses']
    finetune_question+="\n5. characters that will apear in our story : " + characters

    finetune_question+="\nWe are goint to make a central plot of our story. What would be the summary of the story you're going to write in one sentence? With given plot points, imagine a plot that summarizes the whole story you will create in 1-2 sentences. "
    finetune_question+="For example, \"a man goes through several hardships, and finally becomes a god.\""

    # print("\nfor finetune, Question : ")
    # print(finetune_question)
    # print("\nAnswer : ")
    # print(plot)
    from_Plot_points_to_Main_plot.append([{"role":"user", "content":finetune_question},{"role" : "assistant","content":plot}])
    #input()

    ### Scene Information Question Answer
    finetune_question="These are the given plot points. "
    finetune_question+="\n1. the total number of scenes in the story : " + str(num_of_scenes)
    finetune_question+="\n2. emotions that will appear in our story : " + dic_plot_points['emotions']
    finetune_question+="\n3. backgrounds that will appear in our story : " + dic_plot_points['backgrounds']
    finetune_question+="\n4. senses that will appear in our story : " + dic_plot_points['senses']
    finetune_question+="\n5. characters that will apear in our story : " + characters
    finetune_question+="\n6. the summary or overview of our story : " + plot
    finetune_question+="\nNow, let's create the scenes to structure the story. A story is composed of coherent scenes, and a scene is composed of characters, settings, and prevailing emotions and senses. Moreover, each scene should organically intertwine to form a single central plot. Make scenes that contains one or more emotions, backgrounds, senses, and characters from given plot points. Don't make any synopsis of each scene yet. Essentially, use given plot points, but if you really want to add any kind of specific plot points in your scenes for natural flow, you can do this. and there can be unused plot points from given set for the same reason. for example, you would not want to use some characters from the plot points, because their presence seems too sudden or doesn't fit in with the other characters. List scenes one by one with its number and breif information. don't make a specific scene. just give me breif informations of the scene. by end of each scene, you should add a reason why the each scene you wrote is connected to the past scene, starting sentences with the word 'Because'. Especially, you need to explain why each new character is added from the last scene or why each character in the last scene does not appear in current scene, in that cases. Each scene you add to end '_END'. Example of your answer :  \"SCENE 5. Emotion : sadness. Background : ruins, city. Sense : cold and dry. Character : Alex. Explain : Because in the last scene Alex was at the city and he is moving into ruins, the backgrounds are city and ruins.  Because the main background of this scene is ruins, atmosphere is cold and dry. Because Sara did not come with Alex at the last scene, only Alex is main character of this scene. _END.\"."
    
    # print("\nfor finetune, Question : ")
    # print(finetune_question)
    # print("\nAnswer : ")
    # print((' ').join(scenes_informations))
    from_Plot_points_to_Scene_informations.append([{"role":"user", "content":finetune_question},{"role" : "assistant","content":(' ').join(scenes_informations)}])
    #input()


    with open(current_dir+"/dataset/" + jsonl_name, "w", encoding="utf-8") as f:
        # for j in range(len(stories)):
        data = OrderedDict()
        data["message"]=from_Plot_points_to_Characters[-1]
        json.dump(data, f)
        f.write("\n")
        data["message"]=from_Plot_points_to_Main_plot[-1]
        json.dump(data, f)
        f.write("\n")
        data["message"]=from_Plot_points_to_Scene_informations[-1]
        json.dump(data, f)
        f.write("\n")
        data["message"]=from_Scene_informations_to_Events[-1]
        json.dump(data, f)
        f.write("\n")
        for j in range(len(from_Scene_information_to_Scene)):
            data["message"]=from_Scene_information_to_Scene[j]
            json.dump(data, f)
            f.write("\n")

    print(str(total_cost)  + "$")
    return total_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="writingPrompt", action="store")
    parser.add_argument("-model", "--model", default="claude3", action="store")
    parser.add_argument("-few_number", "--few_number", default=100, type=int, action="store")

    args = parser.parse_args()

    maker(args.dataset,args.model, args.few_number)

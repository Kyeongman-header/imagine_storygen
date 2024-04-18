from .generator import *
import argparse
from tqdm import tqdm, trange
from .call import *
import pickle
from nltk.tokenize import word_tokenize
import time
import os
import jsonlines

current_dir = os.path.abspath(__file__)

def story_generation(num=10,finetune=False,gpt_modelname="gpt-3.5-turbo",dataset="None",control_mode=False,
    skip_pe=False,skip_mp=False,skip_sw=False,skip_ce=False,skip_pk=False,skip_cp=False,load_mistral_model_name="None"):
    whole_generations=[]
    fixed_points=[]
    no_fixed=True

    no_use_dataset_do_generate={'pe':False, 'mp' : False, 'sw' : False, 'ce' : False, 'pk' : False, 'cp' : False}
    skip={'pe':skip_pe, 'mp' : skip_mp, 'sw' : skip_sw, 'ce' : skip_ce, 'pk' : skip_pk, 'cp' : skip_cp}

    if "mistral" in gpt_modelname:
        mistral=init_mistral(load_mistral_model_name)
        gpt_modelname=mistral
        caller=call_mistral
    elif "gpt" in gpt_modelname:
        caller=call_openai
    elif "claude" in gpt_modelname:
        caller=call_claude3
    else:
        print("error occur in gpt_modelname. your input:")
        print(gpt_modelname)
        return
    
    if finetune:

        print("use finetuned ver.")
        ids=[]
        temp_suffix=gpt_modelname
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), "finetune_job_id.jsonl"), "r",) as f:
            for line in f:
                ids.append(line)
        for id in ids:
            if gpt_modelname == list(id.keys())[0]:
                gpt_modelname=openai_client.fine_tuning.jobs.retrieve(id[gpt_modelname]).fine_tuned_model
                break
        # print(client.fine_tuning.jobs.retrieve("ftjob-abc123")) # finetune 한 놈의 id를 알아내야 함.
        if temp_suffix==gpt_modelname:
            print("\n\nerror occur.")
            print("There is no " + temp_suffix + " finetune job in finetune_job_id.jsonl.")
            return
        
        print("finetuned_model : ")
        print(gpt_modelname)

    if dataset is not "None":
        # name example => writingPrompts_test
            with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", dataset + "_points.jsonl")) as f:
                for line in f:
                    fixed_points.append(line)
            
            num=len(fixed_points)
            no_fixed=False

            for key, value in no_use_dataset_do_generate.items():
                
                no_use_dataset_do_generate[key]=skip[key]


    story_generator=Story_Generator(finetune,gpt_modelname,caller=caller)
    
    for n in trange(num):
        
        story_generator.reset()
        
        
        #### Plot Point Emergence.

        plot_points={'senses':None if no_fixed or no_use_dataset_do_generate['pe'] else fixed_points[n]['senses'], 'backgrounds':None if no_fixed or no_use_dataset_do_generate['pe'] else fixed_points[n]['backgrounds'], 
        'emotions' : None if no_fixed or no_use_dataset_do_generate['pe'] else fixed_points[n]['emotions'], 'length': None if no_fixed or no_use_dataset_do_generate['pe'] else fixed_points[n]['length'],
        'characters': None if no_fixed or no_use_dataset_do_generate['pe'] else fixed_points[n]['characters']}
        
        if no_fixed is False and no_use_dataset_do_generate['pe'] is False:
            num=word_tokenize(plot_points['length'])
            
            for nu in num:
                if nu.isdigit():
                    plot_points['number']=int(nu)
        

        
        # fixed setting.
        # if you want to ignore this setting, please set 'skip_pe'``
        story_generator.plot_points=plot_points
        
        if (skip_pe is False and no_fixed) or no_use_dataset_do_generate['pe']:
            if control_mode:
                print("\ninput custom length. if you want to skip to set length, just enter.")
                text=input()
                if text!="":
                    if text=="x":
                        plot_points['length']="3, a short story."
                    else:
                        plot_points['length']=text
                    num=word_tokenize(plot_points['length'])
                    for n in num:
                        if n.isdigit():
                            plot_points['number']=int(n)
                print("\ninput custom emotions. if you want to skip to set emotions, just enter.")
                text=input()
                if text!="":
                    if text=="x":
                        plot_points['emotions']="1. Curiosity 2. Fear 3. Relief"
                    else:
                        plot_points['emotions']=text
                
                print("\ninput custom senses. if you want to skip to set senses, just enter.")
                text=input()
                if text!="":
                    if text=="x":
                        plot_points['senses']="1. Icy 2. Aromatic 3. Muffled"
                    else:
                        plot_points['senses']=text
                
                print("\ninput custom backgrounds. if you want to skip to set backgrounds, just enter.")
                text=input()
                if text!="":
                    if text=="x":
                        plot_points['backgrounds']="1. Beach 2. Abandoned factory 3. Enchanted forest"
                    else:
                        plot_points['backgrounds']=text
                
                print("\ninput custom characters. if you want to skip to set characters, just enter.")
                text=input()
                if text!="":
                    if text=="x":
                        plot_points['characters']="1. Eleanor: Female, 58 years old, weathered yet resilient, a former lighthouse keeper with an uncanny ability to predict storms, wears thick glasses with a crack in one lens, has a ritual of walking the coastal village's perimeter daily, harbors a deep-seated guilt for a tragedy at sea she believes she could have prevented, knits tirelessly as a form of meditation. 2. Milo: Male, 32 years old, optimistic but struggling artist with a gentle demeanor, medium build with untidy hair often speckled with paint, finds inspiration in decaying structures and thus spends a lot of time sketching in the abandoned seaside amusement park, wears a locket with a picture of his mother whom he never knew, a mysterious figure led him to this coastal village with promises of capturing a sight no one else can. 3. Cassidy: Non-binary, 24 years old, short with a vibrant tattoo sleeve depicting sea life, owns a small, struggling café in the coastal village that's famous for its peculiar blend of sea salt and caramel coffee, has a soothing voice that contrasts their loud, infectious laugh, secretly feeds stray cats behind the café, moved to the village to escape the cacophony of the city, dreams of turning the café into a sanctuary for artists and misfits."
                    else:
                        plot_points['characters']=text

                print("\nyour custom plot points")
                print(plot_points)

            p=story_generator.Plot_Points_Emergence(plot_points=plot_points)
        
        elif skip_pe:
            plot_points={'senses':"(not given.)", 'backgrounds':"(not given.)", 'emotions' : "(not given.)", 'characters':"(not given.)", 'length' : None}
            if no_fixed :
                p=story_generator.Plot_Points_Emergence(plot_points=plot_points,only_length=True)
            else:
                story_generator.plot_points=plot_points
        

       

        #### Main Plot Maker.
        if no_fixed is False:
            
            story_generator.plot=fixed_points[n]["plot"]

        if (skip_mp is False and no_fixed) or no_use_dataset_do_generate['mp']:
            if control_mode:
                
                print("\ninput custom main plot. if you want to skip to set main plot, just enter.")
                text=input()
                if text!="":
                    if text=='x':
                        story_generator.plot="In a coastal village where the sea whispers secrets, Eleanor, Milo, and Cassidy, each haunted by their pasts, embark on a mysterious journey from a chilling beach through an eerie abandoned factory to an enchanted forest, where they confront their greatest fears and find an unexpected chance at redemption and unity amidst the elements of curiosity, fear, and relief."
                    else:
                        story_generator.plot=text
                else:
                    pl=story_generator.Make_Main_Plot()

                print("\nyour custom plot.")
                print(story_generator.plot)

            else:
                pl=story_generator.Make_Main_Plot()

        elif skip_mp:
            story_generator.plot="(not given.)"
        

        #### Scene weaver.
        if no_fixed is False:
            story_generator.scenes_information=fixed_points[n]["scenes"]
            story_generator.whole_scenes=(' ').join(story_generator.scenes_information)

        if (skip_sw is False and no_fixed) or no_use_dataset_do_generate['sw']:

            s=story_generator.Scene_Weaver()
            

            if control_mode:
                print("\nThese scenes are made my our model. if you want to have any better scenes, please input custom scene information. otherwise, just enter.")
                print("if you want to break loop, enter 'x'.")
                print("if you want to input scene as image, enter 'image'.")

                for i in range(len(story_generator.scenes_information)):
                    print("\nScene number " + str(i+1))
                    print(story_generator.scenes_information[i])
                    text=input("\ninput your custom scene of this scene, or type 'image' if you want to input image. : ")
                    if text!="" and text!="x":
                        if text=="image":
                            url=input("input your custom image url of this scene : ")
                            story_generator.scenes_image[i]=url
                            text=story_generator.Scene_Interpreter(url)
                            

                        story_generator.scenes_information[i]=text
                    elif text=="x":
                        break
                    if i==len(story_generator.scenes_information)-1:
                        i=0

        elif skip_sw:
            story_generator.scenes_information=[]
            for i in range(story_generator.plot_points['number']):
                story_generator.scenes_information.append("(not given.)")
            story_generator.whole_scenes="(not given.)"

        #### Casting on events.

        if no_fixed is False:
            story_generator.events=fixed_points[n]["events"]

        if (skip_ce is False and no_fixed) or no_use_dataset_do_generate['ce']:
            e=story_generator.Casting_On_Events()
            

            if control_mode:
                print("\nThese events are made my our model. if you want to have any better events, please input custom events information. otherwise, just enter.")
                print("if you want to break loop, enter 'x'.")
                for i in range(story_generator.plot_points['number']):
                    print("\nScene number " + str(i))
                    print(story_generator.scenes_information[i])
                    print("event of this scene.")
                    print(story_generator.events[i])
                    text=input("\ninput your custom event of this scene : ")
                    if text!="" and text!="x":
                        story_generator.events[i]=text
                    elif text=="x":
                        break

        elif skip_ce:
            story_generator.events=[]
            for i in range(story_generator.plot_points['number']):
                story_generator.events.append("(not given.)")



        #### Story generation.
        if no_fixed is False:
            story_generator.stories=fixed_points[n]['stories']
            story_generator.summaries=fixed_points[n]['summaries']
            st=fixed_points[n]['stories']
            sm=fixed_points[n]['summaries']
        

        
        if (skip_pk is False and no_fixed) or no_use_dataset_do_generate['pk']:
            
            # no_fixed(dataset 없음)과 skip_pk=False가 동시에 있거나, 혹은 no_use_dataset_do_generate(dataset이 있긴 한데 skip이 주어짐)이면, 
            # generation을 수행한다. 
        
            st,sm=story_generator.Plot_Knitter()
        

        elif skip_pk and no_fixed:
            # no_fixed(dataset없음)과 skip_pk가 True가 동시에 있는 것은 말도 안된다. 이야기를 안 만들겠다는 얘기.
            print("no plot knitter without fixed dataset? this is wrong!")
            return


        if control_mode:
            print("\nThese texts are made my our model. if you want to have any better story, please input custom story. otherwise, just enter.")
            print("if you want to break loop, enter 'x'.")
            for i in range(story_generator.plot_points['number']):
                print("\nScene number " + str(i+1))
                print(story_generator.stories[i])
                print("Summary of this scene.")
                print(story_generator.summaries[i])
                text=input("\ninput your custom story of this scene : ")
                if text!="" and text!="x":
                    
                    words=word_tokenize(text)
                    
                    summary=""
                    story=""

                    for w in words:
                        if w=='SUMMARY':
                            if len(summary)!=0:                                                                        story=summary
                            summary='Scene 1 SUMMARY '
                    else:                         
                        summary+=w+" "
                        
                    
                    story_generator.stories[i]=story
                    story_generator.summaries[i]=summary
                elif text=="x":
                    break
                if i==len(story_generator.events)-1:
                    i=0



        # #### Casting Off Plots.
        if skip_cp is False or no_use_dataset_do_generate['cp']:
            c=story_generator.Casting_Off_Plots()
        else:
            story_generator.past_ver_criticisms=[]
            
        fst=story_generator.stories
        fsm=story_generator.summaries

        
        whole_generations=[{"plot_points" : story_generator.plot_points, "main_plot":story_generator.plot, "events" : story_generator.events, 
        "scenes" : story_generator.scenes_information, "demo_stories" : st, "demo_summaries" : sm, "critisicm" : story_generator.past_ver_criticisms, "final_stories":fst, "final_summaries" : fsm}]
        print(whole_generations)
    
        
        if os.path.exists(current_dir+"results/openai_eval_result.jsonl"):
            mode="a"
        else:
            mode="w"
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), 'results/openai_eval_result.jsonl'), mode="a") as f:
            f.write_all(whole_generations)

        input()
        

# Usage Instruction.

# -finetune ==>
# if you want to use finetuned openai gpt-3.5 turbo, state this.
# and you have to specify -model_name as the suffix(date).  

# -model_name ==> 
# if you want to use the typical LLM openai gpt-3.5-turbo 16K, then just pass.
# if you want to use finetuned version of openai gpt-3.5 turbo, type the suffix(date) of finetuned openai model as mentioned before. 
# if you want to use openai gpt-4, type "gpt-4-0125-preview"
# if you want to use claude 3, type "claude-3-opus-20240229"
# if you want to use finetuned mistral-7B, first you have to run it with at least 2 gpus, and type "mistral", 
# and in that case you have to specify -load_mistral_model_name, as your finetuned model's repo name.

# -dataset ==>    
# if you want to zero-shot generation, then just pass(or type "None.")
# if you want to inject some controllability from specific dataset to the framework, specify dataset names,
# such as "writingPrompts_valid", "writingPrompts_test", "reedsyPrompts_valid", "reedsyPrompts_test", "booksum_valid", "booksum_test"
# or you can load the integrated version, "writingPrompts_reedsyPrompts_booksum_valid" or "..._test".

# -control_mode ==>
# if you want to control manually some stages of DREAMER, state this.
# This is not the same as using dataset. when you check this argument, you will be questioned at each stage to answer your own characters, mood, senses, events, scenes... etc.
# DON'T SET when you using dataset. DREAMER will be confused at some points.

# -num ==>
# if you want generate more than 10 samples, type the numbers.
# but, if you specify dataset(controllability) instead of passing, the input number is ignored and designated as the dataset's length automatically.

# -load_mistral_model_name ==>
# if you designate "-model name" as "mistral", then you have to specify the finetuned model's repo name.
# such as, "Iyan/2024-04-20", etc.

# -skip_pe, -skip_mp, -skip_sw, -skip_ce, -skip_pk, -skip_cp ==>
# skip some stages of DREAMER.
# for zero-shot and fine-tuned model's ablated study(validation of each stage.).

# IT ACTS DIFFERENTLY when you designate dataset(controllability).
# when you experiment controllability, you have to validate the model's capability of reproducing similar writings of humans(like S-bert or ROUGE, BLEU, etc.) when grounded the same plot points.
# so when you designate dataset for the controllability test, skipping means creating by models, not using dataset's plot points.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-finetune", "--finetune", default=False,action="store_true")
    parser.add_argument("-model_name", "--model_name",default="gpt-3.5-turbo-16k", action="store")
    parser.add_argument("-dataset", "--dataset",default="None", action="store")
    parser.add_argument("-control_mode", "--control_mode",default=False, action="store_true")
    parser.add_argument("-num", "--num", type=int, default=10,)
    parser.add_argument("-load_mistral_model_name", "--load_mistral_model_name", default="None",action="store")

    parser.add_argument("-skip_pe", "--skip_pe",default=False, action="store_true")
    parser.add_argument("-skip_mp", "--skip_mp",default=False, action="store_true")
    parser.add_argument("-skip_sw", "--skip_sw",default=False, action="store_true")
    parser.add_argument("-skip_ce", "--skip_ce",default=False, action="store_true")
    parser.add_argument("-skip_pk", "--skip_pk",default=False, action="store_true")
    parser.add_argument("-skip_cp", "--skip_cp",default=False, action="store_true")


    args = parser.parse_args()

    story_generation(num=args.num, finetune=args.finetune,gpt_modelname=args.model_name,dataset=args.dataset,control_mode=args.control_mode,
    skip_pe=args.skip_pe,skip_mp=args.skip_mp,skip_sw=args.skip_sw,skip_ce=args.skip_ce,skip_pk=args.skip_pk,skip_cp=args.skip_cp, load_mistral_model_name=args.load_mistral_model_name)

from .call import *
import pickle
from tqdm import tqdm,trange

class Story_Generator():
    def __init__(self,finetune=False,gpt3_model_name="gpt-3.5-turbo"):
        self.finetune=finetune
        self.gpt3_model_name=gpt3_model_name
        self.past_ver_stories=[]
        self.past_ver_summaries=[]

        if self.finetune:
            self.scenes_information_dataset=None
            self.events_dataset=None
            self.characters_dataset=None
            self.scenes_dataset=None

    def Plot_Points_Emergence(self,):
        self.plot_points={}
        question="You are about to make an amazing story. but before that, you need to decide how long your story will be. Typically, a short story has 3~5 scenes, but a long story has 30~100 or even more. There is no constrains of the length. How many scenes will your story have? answer by number."
        self.plot_points['length']=call_openai(question=question)
        question="List possible emotions 5-10 that can appear in an interesting story. for example, emptiness, sadness, happiness, hatred, fear, wonder, surprise, etc. 
        Don't make any kind of other words. Just list the emotions only, please."
        self.plot_points['emotions']=call_openai(question=question)
        question="List possible backgrounds 5-10 that can appear in an intersting story. for example, a busy city, urban house, school, rural area, mountain, medival castle, etc.
        Don't make any kind of other words. Just list the backgrounds only, please."
        self.plot_points['backgrounds']=call_openai(question=question)
        question="List possible senses 5-10 that can appear in an intersting story. for example, cold, hot, dry, humid, dark, bright, precarious, etc.
        Don't make any kind of other words. Just list the senses only, please."
        self.plot_points['senses']=call_openai(max_tokens=8000,question=question)
        question="A story has some interesting characters. Each character has different characteristics, personality, age, appearance, habits, trauma, growth background, etc. Let’s imagine some characters that will be play roles in our story. 
        We don’t have any specific plot yet, but do have the plot points that will develop into a story. You can freely imagine some characters who are most suitable for our story. 
        Don't limit your imagination to this plot point, but imagine different and diverse characters that might appear in an interesting story. and the plot points are provided below : "
        question+="\nthe total number of scenes in the story : " + self.plot_points['length']
        question+="\nemotions that will appear in our story : " + self.plot_points['emotions']
        question+="\nbackgrounds that will appear in our story : " + self.plot_points['backgrounds']
        question+="\nsenses that will appear in our story : " + self.plot_points['senses']
        question+="\nTaking into account the length of the story, List 1-10 characters you imagined one by one with his or her detailed information. for example, Alex : male, 26 years old, cynical, tall and handsome, a heavy smoker, a hunter, lost his daughter when he was away, raised without parents and experienced war, left-handed, etc."
        
        # The relation between characters is absent here.
        # further research will cover this.

        print(question)

        self.plot_points['characters']=call_openai(max_tokens=2000,question=question,model_name=self.gpt3_model_name)
        
        print(plot_points)

        return plot_points


    def Scene_Weaver(self,):
        
        question="The given plot points : "
        question+="\nthe total number of scenes in the story : " + self.plot_points['length']
        question+="\nemotions that will appear in our story : " + self.plot_points['emotions']
        question+="\nbackgrounds that will appear in our story : " + self.plot_points['backgrounds']
        question+="\nsenses that will appear in our story : " + self.plot_points['senses']
        question+="\ncharacters that will apear in our story : " + self.plot_points['characters']
        question+="\nA story is composed of several scenes, and a scene is composed of characters, settings, and prevailing emotions and senses. 
        Make " + self.plot_points['length'] + " number of scenes that contains one or more emotions, backgrounds, senses, and characters from given plot points. 
        Essentially, use given plot points, but if you really want to add any kind of specific plot points in your scenes for naturalness, you can do this. and there can be unused plot points from given set for the same reason. 
        List scenes one by one with its number and breif information. For example, \"SCENE 1. Emotion : sadness. Background : ruins, city. Sense : cold and dry. Character : Alex. SCENE 2. Emotion :...  \"."
        

        print(question)

        self.whole_scenes=call_openai(max_tokens=10000,question=question,model_name=self.gpt3_model_name)
        print(self.whole_scenes)

        self.scenes_information=[]
        words=word_tokenize(self.whole_scenes)
        scene=""
        for w in words:
            if w=='SCENE':
                if len(scene)!=0:
                    self.scenes_information.append(scene)
                scene='SCENE'
            else:
                scene+=w+" "
        self.scenes_information.append(scene)

        print(self.scenes_information)

        return self.scenes_information

    def Make_What_Happend(self,):
        question="The given scenes of our story, and main characters : "
        question+="\nBreif information of our scenes : " + self.whole_scenes
        question+="\ncharacters that will appear in our story : " + self.plot_points['characters']
        question+="\nWe are almost at the stage right before creating the story! Given the scenes and characters, imagine a plot that summarizes the whole story you will create in 1-2 sentences. "
        question+="For example, \"a man goes through several hardships, and finally becomes a god.\""
        print(question)

        self.plot=call_openai(max_tokens=2000,question=question,model_name=self.gpt3_model_name)
        print(self.plot)

        return self.plot

    def Casting_On_Events(self,):
        question="The given scenes of our story, main characters, and short summary or plot of our whole story : "
        question+="\nBreif information of our scenes : " + self.whole_scenes
        question+="\ncharacters that will appear in our story : " + self.plot_points['characters']
        question+="\nshort summary or plot of our story : " + self.plot
        question+="\nFor each scene in a story, the main characters take several actions for his or her purpose, or for survival, or just because of their habits. Also, there may be some expected or unexpected circumstances which affect and change the characters’ attitude, thoughts, and actions. 
        Those main characters’ actions and expected or unexpected circumstances can be called ‘event’, together. Imagine events that can occur in the i-th scene by considering the given information. Each events should be interesting, unexpected, and impressive. Remember, these events must ultimately lead towards a given plot or short summary of our story."
        question+="\nAnswer this way : SCENE 1. The man is walking through the ruins, but nobody exists there. SCENE 2. Suddenly a girl is watching his back, and the darkness falls. SCENE 3. ..."

        print(question)
        self.whole_events=call_openai(max_tokens=10000,question=question,model_name=self.gpt3_model_name)
        print(self.whole_events)

        self.events=[]
        words=word_tokenize(self.whole_events)
        event=""
        for w in words:
            if w=='SCENE':
                if len(events)!=0:
                    self.events.append(event)
                event='Event '
            else:
                event+=w+" "
        
        self.events.append(event)

        print(self.events)

        return self.events
    
    def plot_knitter_prefix(self,i,past_summarys=""):
        prefix="The given scene information, main characters, and short summary or plot of our whole story : "
        prefix+="\nScene you have to implement : " + self.scenes_information[i] + " " + self.events[i]
        prefix+="\ncharacters that will appear in our story : " + self.plot_points['characters']
        prefix+="\nshort summary or plot of our story : " + self.plot
        prefix+=past_summarys
        return prefix

    def Plot_Knitter(self,):
        
        question="\nWe’re going to make a first scene of a story. There are several information of the current scene. With these things, you should make a corresponding sentences that reflect the information of the scene and main events. 
The sentences’ length should be at least 3 and at most 30 sentences. 
Keep in mind that this is the introduction of the story. It should contain some attractive and mysterious points to attract readers. 
By end of generation, please also give me the short summary of your result by 1~2 sentences. for example, \"SUMMARY : a horrible dog is barking at me.\""
        story=call_openai(max_tokens=5000,self.plot_knitter_prefix(0)+question,model_name=self.gpt3_model_name)
        
        print(story)

        self.summaries=[]
        self.stories=[]
        words=word_tokenize(story)
        summary=""
        for w in words:
            if w=='SUMMARY':
                if len(summary)!=0:
                    self.stories.append(summary)
                summary='Scene 1 SUMMARY '
            else:
                summary+=w+" "
        self.summaries.append(summary)

        print(self.stories)
        print(self.summaries)
        
        
        for j in range(1,int(self.plot_points['length'])):
            primary_question="\nWe’re going to make body of a story. The total number of scenes in our story is "+ self.plot_points['length'] + ", and this is " + str(j+1) +"-th. There are several information of the current and past scenes. With these things, you should make a corresponding sentences that reflect the information of the scene and main events. The sentences’ length should be at least 5 and at most 30 sentences.

Keep in mind that this is the " + str(j+1)+" of " + self.plot_points['length']+ " scenes of the story. It should have progressive, and reasonable sentences that is coherent with the context, but also should be interesting enough that the readers are not sick of our story."



            first_question="\nBefore you generate sentences, you should consider the past generated scenes for coherence. If you need to see the whole text of the past scenes to refresh your memory, please tell me the number of the scene. for example 1, 5, etc.
Do not make the sentences yet. Just tell me the number of the past scenes that you want to look up."

            second_question=primary_question+" By end of generation, please also give me the short summary of your result by 1~2 sentences. for example, \"SUMMARY : a horrible dog is barking at me.\""
            
            if j==int(self.plot_points['length']):
                primary_question="\nWe’re going to make ending of a story. There are several information of the current and past scenes. With these things, you should make a corresponding sentences that reflect the information of the scene and main events. The sentences’ length should be at least 5 and at most 30 sentences.

Keep in mind that this is the end of the story. It should give fascinating climax, or surprising reversal, or appropriate denouement."


            
            past_summarys=(' ').join(self.summaries)
            

            print(self.plot_knitter_prefix(j,past_summary)+primary_question)

            to_see_scenes=call_openai(max_tokens=100, self.plot_knitter_prefix(j,past_summary)+primary_question,model_name=self.gpt3_model_name)
            
            print(to_see_scenes)
            
            words=word_tokenize(to_see_scenes)
            lookup=[]
            for word in words:
                if word.isdigit():
                    lookup.append(int(word))

            past_sentences=('\n').join(['Scene ' + str(i) + " : " + self.stories[i] for i in lookup])

            story=call_openai(max_tokens=5000, self.plot_knitter_prefix(j,past_summary + "\n" + past_sentences)+second_question,model_name=self.gpt3_model_name)
            print(story)

            words=word_tokenize(story)
            summary=""
            for w in words:
                if w=='SUMMARY':
                    if len(summary)!=0:
                        self.stories.append(summary)
                summary='Scene 1 SUMMARY '
            else:
                summary+=w+" "
            self.summaries.append(summary)

            print(self.stories)
            print(self.summaries)

        
        return self.stories, self.summaries

    def Casting_Off_Plots(self,):
        self.criticisms=[]

        past_summarys=self.plot+" "+(' ').join(self.summaries)
        question=past_summarys+"\nWhat can be the narrative arc of our story? Typically, the story has exposition, rising action, climax, falling action, and resolution sequentially.

Before answer the question, please see the short summaries of the scenes, and if you want to see the full text of some specific scenes, give me the numbers of scenes, for example 2, 5, … , etc. Do not answer the question about narrative arc yet. Just tell me the number of the scenes that you want to look up."
        print(question)

        to_see_scenes=call_openai(max_tokens=100, question,model_name=self.gpt3_model_name)
        print(to_see_scenes)

        words=word_tokenize(to_see_scenes)
        lookup=[]
        for word in words:
            if word.isdigit():
                lookup.append(int(word))
        past_sentences=('\n').join(['Scene ' + str(i) + " : " + self.stories[i] for i in lookup])
        
        question=past_summarys+"\n"+past_sentences+"\n"+"Given the information, tell me about each step of narrative arc of this story by considering the characters' circumstances, and gradual disclosure of crucial information. Typically, the story has exposition, rising action, climax, falling action, and resolution sequentially, which called a narrative arc."
        print(question)
        narrative_arc=call_openai(max_tokens=5000, question,model_name=self.gpt3_model_name)
        print(narrative_arc)
        # narrative arc 추출.


        
        self.past_ver_stories.append(self.stories)
        self.past_ver_summaries.append(self.summaries)


        past_summarys+="\nNarrative arc of this story : "+narrative_arc
        

        for j in range(int(self.plot_points['length'])):
            question=past_summarys+' ' + "\nand the current scene is "str(j+1)+" of " + self.plot_points['length']+ " scenes of the whole story. " + "\nScene : " + self.stories[j] +
            "\nNow let's suppose that you are a writer who is making final revisions before facing rigorous critique or judgment. Considering the narrative arc and other contexts, do you believe this scene can function effectively as an integral part of the overall coherent story? If not, please explain your reasoning. 
            before answer, you should see the summaries of other scenes. and if you want to see the whole text of other scenes, please tell me the scene number  for example 2, 5, … , etc. Do not answer the question about revisions yet. Just tell me the number of the other scenes that you want to look up."
            to_see_scenes=call_openai(max_tokens=10000,question,model_name=self.gpt3_model_name)
            print(to_see_scenes)

            words=word_tokenize(to_see_scenes)
            lookup=[]
            for word in words:
                if word.isdigit():
                    lookup.append(int(word))
            past_sentences=('\n').join(['Scene ' + str(i) + " : " + self.stories[i] for i in lookup])
            
            question=past_summarys+"\n"+past_sentences+"\n"+"\nand the current scene is "str(j+1)+" of " + self.plot_points['length']+ " scenes of the whole story. " + "\nScene : " + self.stories[j] + " \n Given the information, Now let's suppose that you are a writer who is making final revisions before facing rigorous critique or judgment. Considering the narrative arc and other various contexts, do you believe this scene can function effectively as an integral part of the overall coherent story? If not, please explain your reasoning."
            
            print(question)
            criticism=call_openai(max_tokens=10000,question,model_name=self.gpt3_model_name)
            print(criticism)
            
            question=past_summarys+"\n"+past_sentences+"\n"+"\nand the current scene is "str(j+1)+" of " + self.plot_points['length']+ " scenes of the whole story. " + "\nScene : " + self.stories[j] + "\n and this is criticism for current scene." + criticism + " \n Based on such criticism, rewrite the scene to make it a more coherent and integral part of the organic narrative arc, and to make the characters more naturally."
            question+=" By end of generation, please also give me the short summary of your result by 1~2 sentences. for example, \"SUMMARY : a horrible dog is barking at me.\""
            print(question)
            rewrite=call_openai(max_tokens=10000,question,model_name=self.gpt3_model_name)
            print(rewrite)
            
            
            words=word_tokenize(rewrite)
            summary=""
            for w in words:
                if w=='SUMMARY':
                    if len(summary)!=0:
                        self.stories[j]=summary
                    summary='Scene' + str(j) + ' SUMMARY '
                else:
                    summary+=w+" "

            self.summaries[j]=summary

            print(self.stories)
            print(self.summaries)
            # rewrite 버전으로 바뀜.
            self.criticisms.append(criticism)
        
        # rewrite.
        
        self.past_ver_criticisms.append(self.criticisms)


    def Decoration(self,):
        # Image를 이용해서 augment하는 건데 마지막에 해보고 성능이 좋아지는지 관찰하자.
        return








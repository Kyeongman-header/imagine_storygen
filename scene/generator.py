from .call import *
import pickle
from tqdm import tqdm,trange
from nltk.tokenize import word_tokenize

class Story_Generator():
    def __init__(self,finetune=False,gpt3_model_name="gpt-3.5-turbo",caller=call_openai):
        self.finetune=finetune
        self.gpt3_model_name=gpt3_model_name
        self.past_ver_stories=[]
        self.past_ver_summaries=[]
        self.past_ver_criticisms=[]
        self.caller=caller

        if self.finetune:
            self.scenes_information_dataset=None
            self.events_dataset=None
            self.characters_dataset=None
            self.scenes_dataset=None


    def reset(self,):
        self.past_ver_stories=[]
        self.past_ver_summaries=[]
        self.past_ver_criticisms=[]

    def Plot_Points_Emergence(self,plot_points=None,only_length=False,):
        print("\n########Plot Points Emergence#########\n")
        if plot_points is None:
            self.plot_points={}
        else:
            self.plot_points=plot_points

        if plot_points['length'] is None:
            question="You are about to make an amazing story. but before that, you need to decide how long your story will be. Typically, a short story has 3~5 scenes, but a long story has 30~100 or even more. There is no constrains of the length. How many scenes will your story have? answer by number with your opinion on whether our story is a long story or a short one. (Only do I need is the number and breif opinion. don't say other things like 'Certainly! Here is a list of possible...' or 'I hope this hepls')"
            self.plot_points['length']=self.caller(question=question,model_name=self.gpt3_model_name)#"gpt-4-0125-preview")
            
            num=word_tokenize(self.plot_points['length'])
            for n in num:
                if n.isdigit():
                    self.plot_points['number']=int(n)
        
        if only_length :
            print("\nAnswer : \n")
            print("length : " + self.plot_points['length'])
            print("senses : " + self.plot_points['senses'])
            print("backgrounds : " + self.plot_points['backgrounds'])
            print("emotions : " + self.plot_points['emotions'])
            print("characters : " + self.plot_points['characters'])
            print("\n")
            return self.plot_points
        
        if plot_points['emotions'] is None:
            question="the total number of scenes in the story : " + self.plot_points['length']
            question+="List possible emotions that can appear in an interesting story. Please remember, don't make too long list. You don't need to make the list as long as the number of scenes. usually there are not more than 2-3 emotions in short stories , so the length of your list as well. but if the story is long, there can be more emotions. Long stories means that it has least 50+ scenes. if we have smaller length of scenes than 50, you should consider our story as a short story. and remember, these emotions will appear in one story! In a short story, emotions that are too disconnected from each other find it challenging to coexist. Your answer example : 1. emptiness, 2. sadness, 3. happiness, 4. hatred, 5. fear, 6. wonder, 7. surprise, etc. Don't make any kind of other words. Just list the emotions only, please.(Only do I need is the list of emotions. don't say other things like 'Certainly! Here is a list of possible...' or 'I hope this hepls')"
            self.plot_points['emotions']=self.caller(question=question, model_name=self.gpt3_model_name)#"gpt-4-0125-preview")
        
        if plot_points['backgrounds'] is None:
            question="the total number of scenes in the story : " + self.plot_points['length']
            question+="List possible backgrounds that can appear in an intersting story. Please remember, don't make too long list. You don't need to make the list as long as the number of scenes. usually there are not more than 2-3 backgrounds in short stories, so the length of your list as well. but if the story is long, there can be more backgrounds. Long stories means that it has least 50+ scenes. if we have smaller length of scenes than 50, you should consider our story as a short story. and remember, these backgrounds will appear in one story! In a short story, backgrounds that are too disconnected from each other find it challenging to coexist. Your answer example : 1. a busy city, 2. urban house, 3. school, 4. rural area, 5. mountain, 6. medival castle, etc. Don't make any kind of other words. Just list the backgrounds only, please.(Only do I need is the list of backgrounds. don't say other things like 'Certainly! Here is a list of possible...' or 'I hope this hepls')"
            self.plot_points['backgrounds']=self.caller(question=question, model_name=self.gpt3_model_name)#"gpt-4-0125-preview")
        
        if plot_points['senses'] is None:
            question="the total number of scenes in the story : " + self.plot_points['length']
            question+="List possible senses that can appear in an intersting story. Please remember, don't make too long list. You don't need to make the list as long as the number of scenes. usually there are not more than 2-3 senses in short stories, so the length of your list as well. but if the story is long, there can be more senses. Long stories means that it has least 50+ scenes. if we have smaller length of scenes than 50, you should consider our story as a short story. and remember, these senses will appear in one story! In a short story, senses that are too disconnected from each other find it challenging to coexist. Your answer example : 1. cold, 2. dry, 3. dark, etc. Don't make any kind of other words. Just list the senses only, please.(Only do I need is the list of senses. don't say other things like 'Certainly! Here is a list of possible...' or 'I hope this hepls')"
            self.plot_points['senses']=self.caller(question=question,model_name=self.gpt3_model_name)#"gpt-4-0125-preview")
        
        if plot_points['characters'] is None:
            question="A story has some interesting characters. Each character has different characteristics, personality, age, appearance, habits, trauma, growth background, etc. Let’s imagine some characters that will be play roles in our story. We don’t have any specific plot yet, but do have the plot points that will develop into a story. You can freely imagine some characters who are most suitable for our story. Don't limit your imagination to this plot point, but imagine different and diverse characters that might appear in an interesting story. The plot points are provided below. "
            question+="\n1. the total number of scenes in the story : " + self.plot_points['length']
            question+="\n2. emotions that will appear in our story : " + self.plot_points['emotions']
            question+="\n3. backgrounds that will appear in our story : " + self.plot_points['backgrounds']
            question+="\n4. senses that will appear in our story : " + self.plot_points['senses']
            question+="\nTaking into account the length of the story, List some characters you imagined one by one with his or her detailed information. Please remember, don't make too long list. You don't need to make the list as long as the number of scenes. usually there are not more than 2-3 characters in short stories, so the length of your list as well. But if the story is long, there can be more than 3 characters. Long stories means that it has least 50+ scenes. if we have smaller length of scenes than 50, you should consider our story as a short story. and remember, these characters will appear in one story! In a short story, characters that are too disconnected from each other find it challenging to coexist. your answer example: Alex : male, 26 years old, cynical, tall and handsome, a heavy smoker, a hunter, lost his daughter when he was away, raised without parents and experienced war, left-handed, etc.(Only do I need is the list of characters. don't say other things like 'Certainly! Here is a list of possible...' or 'I hope this hepls!')"


        # The relation between characters is absent here.
        # further research will cover this.

            #print(question)

            self.plot_points['characters']=self.caller(question=question,model_name=self.gpt3_model_name)#"gpt-4-0125-preview")
        
        
        print("\nAnswer : \n")
        print("length : " + self.plot_points['length'])
        print("senses : " + self.plot_points['senses'])
        print("backgrounds : " + self.plot_points['backgrounds'])
        print("emotions : " + self.plot_points['emotions'])
        print("characters : " + self.plot_points['characters'])
        print("\n")
        return self.plot_points

        
    def Scene_Interpreter(self,url):
        question="Look at this image. what do you feel? what are there in the image? Who is there? Answer these questions by listing emotion, sense, background, and character you can observe from the image. For example,  \"SCENE. Emotion : sadness. Background : ruins, city. Sense : cold and dry. Character : Alex. ...\""
        question+="and remember that if there is someone in the image, he or she or it is one of these people : " + self.plot_points['characters']
        print(question)
        scene_information=self.caller(max_tokens=10000,question=question,model_name=self.gpt3_model_name,multimodal=url)
        print(scene_information)
        return scene_information

    def Scene_Weaver(self,):
        print("\n########Scene Weaver#########\n")
        self.whole_scenes=""

        p_question="These are the given plot points. "
        p_question+="\n1. the total number of scenes in the story : " + self.plot_points['length']
        p_question+="\n2. emotions that will appear in our story : " + self.plot_points['emotions']
        p_question+="\n3. backgrounds that will appear in our story : " + self.plot_points['backgrounds']
        p_question+="\n4. senses that will appear in our story : " + self.plot_points['senses']
        p_question+="\n5. characters that will apear in our story : " + self.plot_points['characters']
        p_question+="\n6. the summary or overview of our story : " + self.plot
        """
        for j in range(1,int(self.plot_points['number']+1)): 
            if j!=1:
                question="\nand the past scenes : \n"
                question+=self.whole_scenes
            else:
                question=""
            question+="\nNow, let's create the " + str(j) +  "-th scene to structure the story. A story is composed of coherent scenes, and a scene is composed of characters, settings, and prevailing emotions and senses. Moreover, each scene should organically intertwine to form a single central plot. Make a scene that contains one or more emotions, backgrounds, senses, and characters from given plot points. Don't make any synopsis of each scene yet. Essentially, use given plot points, but if you really want to add any kind of specific plot points in your scenes for natural flow, you can do this. and there can be unused plot points from given set for the same reason. for example, you would not want to use some characters from the plot points, because their presence seems too sudden or doesn't fit in with the other characters. List scenes one by one with its number and breif information. For example, \"SCENE " + str(j) + ". Emotion : sadness. Background : ruins, city. Sense : cold and dry. Character : Alex. \". don't make a specific scene. just give me breif information of the scene. Don't make the whole scenes of our story at once. Just give me the " + str(j) + "-th scene information."
            
            print(question)
            scene=self.caller(max_tokens=10000,question=p_question + question,model_name=self.gpt3_model_name) + " "
            self.whole_scenes+=scene + " "
            print(scene)
        #print(self.whole_scenes)
        #input()
        """

        question="\nNow, let's create the scenes to structure the story. A story is composed of coherent scenes, and a scene is composed of characters, settings, and prevailing emotions and senses. Moreover, each scene should organically intertwine to form a single central plot. Make scenes that contains one or more emotions, backgrounds, senses, and characters from given plot points. Don't make any synopsis of each scene yet. Essentially, use given plot points, but if you really want to add any kind of specific plot points in your scenes for natural flow, you can do this. and there can be unused plot points from given set for the same reason. for example, you would not want to use some characters from the plot points, because their presence seems too sudden or doesn't fit in with the other characters. List scenes one by one with its number and breif information. don't make a specific scene. just give me breif informations of the scene. by end of each scene, you should add a reason why the each scene you wrote is connected to the past scene, starting sentences with the word 'Because'. Especially, you need to explain why each new character is added from the last scene or why each character in the last scene does not appear in current scene, in that cases. Each scene you add to end '_END'. Example of your answer :  \"SCENE 5. Emotion : sadness. Background : ruins, city. Sense : cold and dry. Character : Alex. Explain : Because in the last scene Alex was at the city and he is moving into ruins, the backgrounds are city and ruins.  Because the main background of this scene is ruins, atmosphere is cold and dry. Because Sara did not come with Alex at the last scene, only Alex is main character of this scene. _END.\"."
        print("\nQuestion : ")
        print(question)
        scene=self.caller(question=p_question + question,model_name=self.gpt3_model_name)
        self.whole_scenes=scene
        #print(scene)


        self.scenes_information=[]
        words=word_tokenize(self.whole_scenes)
        #print(words)
        scene=""
        for w in words:
            if w=='_END':
                if len(scene)!=0:
                    self.scenes_information.append(scene)
                #scene='SCENE'
                scene=""
            else:
                scene+=w+" "
        #self.scenes_information.append(scene)
        
        print("\nAnswer : ")
        print(self.scenes_information)
        print(len(self.scenes_information))

        self.scenes_image=[]
        for i in range(len(self.scenes_information)):
            self.scenes_image.append(None)

        return self.scenes_information

    def Make_Main_Plot(self,):
        print("\n########Make Main Plot#########\n")

        question="These are the given plot points."
        question+="\n1. the total number of scenes in the story : " + self.plot_points['length']
        question+="\n2. emotions that will appear in our story : " + self.plot_points['emotions']
        question+="\n3. backgrounds that will appear in our story : " + self.plot_points['backgrounds']
        question+="\n4. senses that will appear in our story : " + self.plot_points['senses']
        question+="\n5. characters that will apear in our story : " + self.plot_points['characters']
        
        question+="\nWe are goint to make a central plot of our story. What would be the summary of the story you're going to write in one sentence? With given plot points, imagine a plot that summarizes the whole story you will create in 1-2 sentences. "
        question+="For example, \"a man goes through several hardships, and finally becomes a god.\""
        
        print("\nQuestion : ")
        print(question)

        self.plot=self.caller(max_tokens=2000,question=question,model_name=self.gpt3_model_name)
        print("\nAnswer : ")
        print(self.plot)

        return self.plot

    def Casting_On_Events(self,):
        print("\n########Casting ON Events.#########\n")


        question="These are the given scenes of our story, main characters, and short summary or whole plot of our whole story."
        question+="\n1. Breif summaries of our scenes : " + self.whole_scenes
        question+="\n2. characters that will appear in our story : " + self.plot_points['characters']
        question+="\n3. short summary or over view of our total story : " + self.plot
        question+="\nFor each scene in a story, the main characters take several actions for his or her purpose, or for survival, or just because of their habits. Also, there may be some expected or unexpected circumstances which affect and change the characters’ attitude, thoughts, and actions. Those main characters’ actions and expected or unexpected circumstances can be called ‘event’, together. Imagine events that can occur in the i-th scene by considering the given information. Each events should be interesting, unexpected, and impressive. Remember, these events must ultimately lead towards a given plot or short summary of our story."
        question+=" by end of each scene, you should add a reason why each events you wrote is connected to the past scene and is part of the whole plot, starting sentences with the word 'Because'. Each scene you add to end '_END'."
        question+="\nAnswer this way : SCENE 1. The man is walking through the ruins, but nobody exists there. Explain : Because this is the first scene of our story, this event can be the attractive introduction. _END.  SCENE 2. Suddenly a mysterious girl is watching his back, and the darkness falls. Explain : Because at the last scene the man was walking through the ruins, appearance of a mysterious girl is suitable for an intersting story and the whole plot. _END. SCENE 3. ..."



        print("\nQuestion :")
        print(question)
        self.whole_events=self.caller(question=question,model_name=self.gpt3_model_name)
        #print(self.whole_events)
        #input()

        self.events=[]
        words=word_tokenize(self.whole_events)
        event=""
        for w in words:
            if w=='_END':
                if len(event)!=0:
                    self.events.append(event)
                event=""
            else:
                event+=w+" "
        
        #self.events.append(event)
        print("\nAnswer : ")
        print(self.events)
        print(len(self.events))

        return self.events
    
    def plot_knitter_prefix(self,i,past_summarys=""):

        

        prefix="These are informations about current scene you need to write."
        prefix+="\n1. short summary or overview of our story : " + self.plot
        prefix+="\n2. characters that will appear in our whole story :  " + self.plot_points['characters']
        prefix+="\n3. Scene you have to implement : " + self.scenes_information[i] + " " + self.events[i]

        
        prefix+="\n" + past_summarys + "\n"
        return prefix

    def Plot_Knitter(self,):
        
        print("\n########Plot Knitter#########\n")

        question="\nWe’re going to make a first scene of a story. There are several information of the current scene. With these things, you should make a corresponding sentences that reflect the information of the scene and main events. The sentences’ length should be at least 3 and at most 30 sentences. Keep in mind that this is the introduction of the story. It should contain some attractive and mysterious points to attract readers. By end of generation, please also give me the short summary of your result by 1~2 sentences, starting with the word '_SUMMARY : '. and add to end of your total generation '_END'. for example, \"... #_SUMMARY : a horrible dog is barking at me. _END.\""
        
        print("\nQuestion : ")
        print(self.plot_knitter_prefix(0)+question)
        story=self.caller(question=self.plot_knitter_prefix(0)+question,model_name=self.gpt3_model_name,multimodal=self.scenes_image[0],add_image_explain=True)
        print("\nAnswer :")
        
        
        
        

        self.summaries=[]
        self.stories=[]
        words=word_tokenize(story)
        summary=""
        for w in words:
            if w=='_SUMMARY' :
                if len(summary)!=0:
                    self.stories.append(summary)
                summary='Scene 1 SUMMARY '
            else:
                summary+=w+" "
        self.summaries.append(summary)

        print(self.stories)
        print(self.summaries)
        
        
        for j in range(1,int(self.plot_points['number'])):
            primary_question="\nWe’re going to make body of a story. This is the " + str(j+1) +"-th scene of the total " + str(self.plot_points['number']) + "scenes in our story. As you can see, there are several information of the current and past scenes. With these things, you should make a corresponding sentences that reflect the information of the scene and main events. The sentences’ length should be at least 5 and at most 30 sentences. Keep in mind that this is the " + str(j+1)+" of " + str(self.plot_points['number']) + " scenes of the story, it should have progressive, and reasonable sentences that is coherent with the context, but also should be interesting enough that the readers are not sick of our story."
            
            
            
            
            
            if j==int(self.plot_points['number'])-1:
                primary_question="\nWe’re going to make ending of a story. This is the " + str(j+1) +"-th scene of the total " + str(self.plot_points['number']) + "scenes in our story. \nAs you can see, there are several information of the current and past scenes. With these things, you should make a corresponding sentences that reflect the information of the scene and main events. The sentences’ length should be at least 5 and at most 30 sentences. Keep in mind that this is the end of the story. It should give fascinating climax, or surprising reversal, or appropriate denouement."
            primary_question+="\nand the last generated scene is given by default."
            primary_question+="\nScene #" + str(j) + ". " + self.stories[j-1]
            
            past_summarys=('\n').join(self.summaries)
            
            
            first_question="\nBefore you generate sentences, you should consider the past generated scenes for coherence. If you need to see the whole text of the past scenes to refresh your memory, please tell me the numbers of the past scenes. For example, your answer : \"Scene 1, Scene 5.\" Attention! Never, never, Do not say anything but answer the scene numbers you need to see. Only if there is no past scenes you would like to see, answer \"Nothing.\" Please keep in mind this."

            print("\nQuestion : ")
            print(self.plot_knitter_prefix(j,past_summarys)+primary_question+first_question)
            
            to_see_scenes=self.caller(question=self.plot_knitter_prefix(j,past_summarys)+primary_question + first_question,model_name=self.gpt3_model_name,multimodal=self.scenes_image[j])
            print("\nAnswer : " )
            print(to_see_scenes)
            

            words=word_tokenize(to_see_scenes)
            lookup=[]
            for word in words:
                if word.isdigit():
                    lookup.append(int(word)-1)

            past_sentences=('\n').join(['Scene ' + str(i) + " : " + self.stories[i] for i in lookup])
            
            if len(lookup)!=0:
                past_sentences="\nThese are past scenes you wanted to see. Please refer to these and write consistently.\n" + past_sentences
            
            second_question=" By end of generation, please also give me the short summary of your result by 1~2 sentences, starting with the word '_SUMMARY : '.  and add to end of your total generation '_END'. for example, \"... _SUMMARY : a horrible dog is barking at me. _END.\""

            print("\nQuestion : ")
            print(self.plot_knitter_prefix(j,past_summarys + "\n" + past_sentences) + primary_question + second_question)

            story=self.caller(question=self.plot_knitter_prefix(j,past_summarys + "\n" + past_sentences) + primary_question + second_question,model_name=self.gpt3_model_name,multimodal=self.scenes_image[j])
            
            print("\nAnswer : ")
            print(story)
            

            words=word_tokenize(story)
            summary=""
            for w in words:
                if w=='_SUMMARY':
                    if len(summary)!=0:
                        self.stories.append(summary)
                    summary='Scene ' + str(j+1)+ ' SUMMARY. '
                else:
                    summary+=w+" "
            self.summaries.append(summary)
            
            print("\n total stories and summaries")
            print(self.stories)
            print(self.summaries)

        
        return self.stories, self.summaries

    def Casting_Off_Plots(self,):

        print("\n########Casting Off Plots#########\n")

        self.criticisms=[]

        past_summarys="The whole plot of our story : " + self.plot+"\nand summaries of each scenes of our story : "+('\n').join(self.summaries)
        question=past_summarys+"\nWhat can be the narrative arc of our story? Typically, the story has exposition, rising action, climax, falling action, and resolution sequentially. Before answer the question, please see the short summaries of the scenes, and if you want to see the full text of some specific scenes, give me the numbers of scenes, for example 2, 5, … , etc. Do not answer the question about narrative arc yet. Just tell me the number of the scenes that you want to look up.  Attention! Never, never, Do not say anything but answer the scene numbers you need to see. Only if there is no past scenes you would like to see, answer \"Nothing.\" Please keep in mind this."
        print("\nQuestion : ")
        print(question)

        to_see_scenes=self.caller(max_tokens=100, question=question,model_name=self.gpt3_model_name)
        print("\nAnswer : ")
        print(to_see_scenes)
        

        words=word_tokenize(to_see_scenes)
        lookup=[]
        for word in words:
            if word.isdigit():
                lookup.append(int(word)-1)
        past_sentences=('\n').join(['Scene ' + str(i) + " : " + self.stories[i] for i in lookup])
        if len(lookup)!=0:
                past_sentences="\nand these are past scenes you wanted to see. Please refer to these and write consistently.\n" + past_sentences
        question=past_summarys+past_sentences+"\nGiven the information, tell me about each step of narrative arc of this story by considering the characters' circumstances, and gradual disclosure of crucial information. Typically, the story has exposition, rising action, climax, falling action, and resolution sequentially, which called a narrative arc."
        print("\nQuestion : ")
        print(question)
        narrative_arc=self.caller(question=question,model_name=self.gpt3_model_name)
        print("\nAnswer : ")
        print(narrative_arc)
        # narrative arc 추출.


        
        self.past_ver_stories.append(self.stories)
        self.past_ver_summaries.append(self.summaries)


        past_summarys+="\nNarrative arc of this story : "+narrative_arc
        

        for j in range(int(self.plot_points['number'])):
            question=past_summarys+"\nand this is the "+ str(j+1)+"-th scene of " + str(self.plot_points['number']) + " total scenes of the whole story. " + "\nScene : " + self.stories[j] + "\nNow let's suppose that you are a writer who is making final revisions before facing rigorous critique or judgment. Considering the narrative arc and other contexts, do you believe this scene can function effectively as an integral part of the overall coherent story? If not, please explain your reasoning. before answer, you should see the summaries of other scenes. and if you want to see the whole text of other scenes, please tell me the scene number. Attention! Never, never, Do not say anything but answer the scene numbers you need to see. Only if there is no past scenes you would like to see, answer \"Nothing.\" Please keep in mind this. For example 2, 5, … , etc. Do not answer the question about revisions yet. Just tell me the number of the other scenes that you want to look up."
            
            print("\nQuestion : ")
            print(question)
            to_see_scenes=self.caller(question=question,model_name=self.gpt3_model_name)
            
            print("\nAnswer :")
            print(to_see_scenes)
            

            words=word_tokenize(to_see_scenes)
            lookup=[]
            for word in words:
                if word.isdigit():
                    lookup.append(int(word)-1)
            past_sentences=('\n').join(['Scene ' + str(i+1) + " : " + self.stories[i] for i in lookup])
            if len(lookup)!=0:
                past_sentences="\nThese are past scenes you wanted to see. Please refer to these and write consistently.\n" + past_sentences 
            question=past_summarys+"\n"+past_sentences+"\n"+"and the current scene is "+ str(j+1)+" of " + str(self.plot_points['number'])+ " scenes of the whole story. " + "\nScene : " + self.stories[j] + " \n Given the information, Now let's suppose that you are a writer who is making final revisions before facing rigorous critique or judgment. Considering the narrative arc and other various contexts, do you believe this scene can function effectively as an integral part of the overall coherent story? If not, please explain your reasoning."
            print("\nQuestion : ")
            print(question)
            criticism=self.caller(question=question,model_name=self.gpt3_model_name)
            print("\nAnswer : ")
            print(criticism)
            
            question=past_summarys+past_sentences+"\nand the current scene is "+ str(j+1)+" of " + str(self.plot_points['number'])+ " scenes of the whole story. \nScene : " + self.stories[j] + "\n and this is criticism for current scene : \n" + criticism + " \n Based on such criticism, rewrite the scene to make it a more coherent and integral part of the organic narrative arc, and to make the characters more naturally."
            question+=" By end of generation, please also give me the short summary of your result by 1~2 sentences, starting with the word '_SUMMARY : '. and add to end of your total generation '_END'. for example, \"... _SUMMARY : a horrible dog is barking at me. _END.\""
            
            print("\nQuestion : ")
            print(question)
            rewrite=self.caller(question=question,model_name=self.gpt3_model_name)
            print("\nAnswer : ")
            print(rewrite)
            

            words=word_tokenize(rewrite)
            summary=""
            for w in words:
                if w=='_SUMMARY':
                    if len(summary)!=0:
                        self.stories[j]=summary
                    summary='Scene' + str(j+1) + ' SUMMARY '
                else:
                    summary+=w+" "

            self.summaries[j]=summary
            
            print("\n whole rewrited stories and summaries")
            print(self.stories)
            print(self.summaries)
            # rewrite 버전으로 바뀜.
            self.criticisms.append(criticism)
            

        # rewrite.
        
        self.past_ver_criticisms.append(self.criticisms)
        return self.past_ver_criticisms

    def Decoration(self,):
        # Image를 이용해서 augment하는 건데 마지막에 해보고 성능이 좋아지는지 관찰하자.
        return








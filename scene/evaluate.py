import os
import argparse
import jsonlines
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

current_dir = os.path.abspath(__file__)




def SBERT(datasets):
    sentence_bert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",device="cuda")
    avg_SBERT_score=0
    whole_count=0


    for lines in datasets:
        for line in lines:
            pred=sentence_bert.encode((' ').join(line['final_stories']))
            label=sentence_bert.encode((' ').join(line['golden_label_stories']))
            c=util.cos_sim(pred,label).item()
            avg_SBERT_score+=c
            whole_count+=1
    
    avg_SBERT_score=avg_SBERT_score/whole_count
    print(avg_SBERT_score)
    
    return {"SBERT" : avg_SBERT_score}


def ROUGE(datasets):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    whole_count=0

    rouge={ "rouge1" :0, "rouge2" :0, "rougeL" : 0}

    for lines in datasets:
        for line in lines:
            scores=scorer.score((' ').join(line['final_stories']),(' ').join(line['golden_label_stories']))
            whole_count+=1
            for key in scores:
                rouge[key] += scores[key]
    
    for key in list(rouge.keys()):
        rouge[key]=rouge[key]/whole_count
    
    print(rouge)

    return rouge
    


def BLUE(datasets):
    whole_count=0

    bleu={ "bleu1" :0, "bleu2" :0, "bleu3" : 0, "bleu4" : 0}

    for lines in datasets:
        for line in lines:
            candidate=word_tokenize((' ').join(line['final_stories']))
            reference=[word_tokenize((' ').join(line['golden_label_stories']))]
            bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu(reference, candidate, weights=(1./2., 1./2., 0, 0))
            bleu3 = sentence_bleu(reference, candidate, weights=(1./3., 1./3., 1./3., 0))
            bleu4 = sentence_bleu(reference, candidate, weights=(1./4., 1./4., 1./4., 1./4.))
            bleu["bleu1"]+=bleu1
            bleu["bleu2"]+=bleu2
            bleu["bleu3"]+=bleu3
            bleu["bleu4"]+=bleu4
            whole_count +=1
    
    for key in list(bleu.keys()):
        bleu[key]=bleu[key]/whole_count
    
    print(bleu)

    return bleu


def calculater(a_story):
        bleu={ "in_self_bleu1" :0, "in_self_bleu2" :0, "in_self_bleu3" : 0, "in_self_bleu4" : 0}
        for j in len(a_story):
                candidate=word_tokenize(a_story[j])
                references=[[word_tokenize(story)] for story in a_story]
                references.pop(j)
                bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
                bleu2 = sentence_bleu(references, candidate, weights=(1./2., 1./2., 0, 0))
                bleu3 = sentence_bleu(references, candidate, weights=(1./3., 1./3., 1./3., 0))
                bleu4 = sentence_bleu(references, candidate, weights=(1./4., 1./4., 1./4., 1./4.))
                bleu["in_self_bleu1"]+=bleu1
                bleu["in_self_bleu2"]+=bleu2
                bleu["in_self_bleu3"]+=bleu3
                bleu["in_self_bleu4"]+=bleu4
        
        for key in list(bleu.keys()):
            bleu[key]=bleu[key]/len(a_story)
        
        return bleu

def in_Self_BLEU(datasets):
    whole_count=0

    bleu={ "in_self_bleu1" :0, "in_self_bleu2" :0, "in_self_bleu3" : 0, "in_self_bleu4" : 0}
    golden_bleu={"in_self_bleu1" : 0, "in_self_bleu2" :0, "in_self_bleu3" : 0, "in_self_bleu4" : 0}

    for lines in datasets:
        for line in lines:
                p=calculator(line['final_stories'])
                g=calculator(line['golden_label_stories'])
                for key in list(bleu.keys()):
                    bleu[key]+=p[key]
                    golden_bleu[key]+=g[key]
                
                whole_count +=1
    
    for key in list(bleu.keys()):
        bleu[key]=bleu[key]/whole_count
        golden_bleu[key]=golden_bleu[key]/whole_count
    
    rename={"in_self_bleu1" : "golden_in_self_bleu1", "in_self_bleu2" : "golden_in_self_bleu2", 
    "in_self_bleu3" : "golden_in_self_bleu3", "in_self_bleu4" : "golden_in_self_bleu4"}
    golden_bleu=dict((rename[key], value) for (key, value) in golden_bleu.items())
    
    print(bleu)
    print(golden_bleu)

    return bleu.update(golden_bleu)

def in_Self_SBERT(datasets):
    whole_count=0

    sentence_bert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",device="cuda")
    in_self_SBERT_score=0
    golden_in_self_SBERT_score=0
    
    whole_count=0

    for lines in datasets:
        for line in lines:
            cc=0
            for j in len(line['final_stories']):
                candidate=line['final_stories']
                references=[story for story in line['final_stories']]
                references.pop(j)
                pred=sentence_bert.encode(candidate)
                labels=[sentence_bert.encode(golden_story) for golden_story in references]
                cs=[util.cos_sim(pred,label).item() for label in labels]
                cc+=sum(cs)/len(cs)
            
            cc=cc/len(line['final_stories'])
            in_self_SBERT_score+=cc

            cc=0
            for j in len(line['golden_label_stories']):
                candidate=line['golden_label_stories']
                references=[story for story in line['golden_label_stories']]
                references.pop(j)
                pred=sentence_bert.encode(candidate)
                labels=[sentence_bert.encode(golden_story) for golden_story in references]
                cs=[util.cos_sim(pred,label).item() for label in labels]
                cc+=sum(cs)/len(cs)
            
            cc=cc/len(line['golden_label_stories'])
            golden_in_self_SBERT_score+=cc
            whole_count +=1

    in_self_SBERT_score=in_self_SBERT_score/whole_count
    golden_in_self_SBERT_score=golden_in_self_SBERT_score/whole_count
    
    print(in_self_SBERT_score)
    print(golden_in_self_SBERT_score)

    return {"in_self_SBERT_score" : in_self_SBERT_score, "golden_in_self_SBERT_score" :golden_in_self_SBERT_score}


def Self_BLEU(datasets):
    whole_count=0

    bleu={ "in_self_bleu1" :0, "in_self_bleu2" :0, "in_self_bleu3" : 0, "in_self_bleu4" : 0}
    golden_bleu={"in_self_bleu1" : 0, "in_self_bleu2" :0, "in_self_bleu3" : 0, "in_self_bleu4" : 0}
    
    for lines in datasets:
        for j in len(lines):
            candidate=word_tokenize((' ').join(lines[j]['final_stories']))
            references=[[word_tokenize((' ').join(data['final_stories']))] for data in lines]
            references.pop(j)
            bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu(references, candidate, weights=(1./2., 1./2., 0, 0))
            bleu3 = sentence_bleu(references, candidate, weights=(1./3., 1./3., 1./3., 0))
            bleu4 = sentence_bleu(references, candidate, weights=(1./4., 1./4., 1./4., 1./4.))
            bleu["in_self_bleu1"]+=bleu1
            bleu["in_self_bleu2"]+=bleu2
            bleu["in_self_bleu3"]+=bleu3
            bleu["in_self_bleu4"]+=bleu4

            candidate=word_tokenize((' ').join(lines[j]['golden_label_stories']))
            references=[[word_tokenize((' ').join(data['golden_label_stories']))] for data in lines]
            references.pop(j)
            bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu(references, candidate, weights=(1./2., 1./2., 0, 0))
            bleu3 = sentence_bleu(references, candidate, weights=(1./3., 1./3., 1./3., 0))
            bleu4 = sentence_bleu(references, candidate, weights=(1./4., 1./4., 1./4., 1./4.))
            golden_label_bleu["in_self_bleu1"]+=bleu1
            golden_label_bleu["in_self_bleu2"]+=bleu2
            golden_label_bleu["in_self_bleu3"]+=bleu3
            golden_label_bleu["in_self_bleu4"]+=bleu4

            whole_count +=1
    
    for key in list(bleu.keys()):
        bleu[key]=bleu[key]/whole_count
        golden_bleu[key]=golden_bleu[key]/whole_count
    
    rename={"in_self_bleu1" : "golden_in_self_bleu1", "in_self_bleu2" : "golden_in_self_bleu2", 
    "in_self_bleu3" : "golden_in_self_bleu3", "in_self_bleu4" : "golden_in_self_bleu4"}
    golden_bleu=dict((rename[key], value) for (key, value) in golden_bleu.items())
    
    print(bleu)
    print(golden_bleu)
    return bleu.update(golden_bleu)


def Self_SBERT(datasets):
    whole_count=0

    sentence_bert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",device="cuda")
    in_self_SBERT_score=0
    golden_in_self_SBERT_score=0


    for lines in datasets:
        for j in len(lines):
            candidate=(' ').join(lines[j]['final_stories'])
            references=[[(' ').join(data['final_stories'])] for data in lines]
            references.pop(j)
            pred=sentence_bert.encode(candidate)
            labels=[sentence_bert.encode(golden_story) for golden_story in references]
            cs=[util.cos_sim(pred,label).item() for label in labels]
            in_self_SBERT_score+=sum(cs)/len(cs)

            candidate=(' ').join(lines[j]['golden_label_stories'])
            references=[[(' ').join(data['golden_label_stories'])] for data in lines]
            references.pop(j)
            pred=sentence_bert.encode(candidate)
            labels=[sentence_bert.encode(golden_story) for golden_story in references]
            cs=[util.cos_sim(pred,label).item() for label in labels]
            golden_in_self_SBERT_score+=sum(cs)/len(cs)
            
            whole_count+=1


    in_self_SBERT_score=in_self_SBERT_score/whole_count
    golden_in_self_SBERT_score=golden_in_self_SBERT_score/whole_count
    
    print(in_self_SBERT_score)
    print(golden_in_self_SBERT_score)

    return {"in_self_SBERT_score":in_self_SBERT_score,"golden_in_self_SBERT_score":golden_in_self_SBERT_score}
# Usage Instruction.

# SBERT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-text_file", "--text_file", default="test",action="store")

    # golden label evaluation (automatic).
    parser.add_argument("-SBERT","--SBERT", default=True,action="store_true")
    parser.add_argument("-ROUGE","--ROUGE", default=True,action="store_true")
    parser.add_argument("-BLEU","--BLEU", default=True,action="store_true")

    # repetitiveness test. (automatic).
    parser.add_argument("-in_Self_BLEU","--in_Self_BLEU",default=True,action="store_true")
    parser.add_argument("-in_Self_SBERT","--in_Self_SBERT",default=True,action="store_true")
    parser.add_argument("-Self_BLEU","--Self_BLEU",default=True,action="store_true")
    parser.add_argument("-Self_SBERT","--Self_SBERT",default=True,action="store_true")

    

    # comparision between model evaluation (human or LLM).
    # We made a survey website for these metrics.
    
    # parser.add_argument("-Creativity","--Creativity", default=True,action="store_true")
    # parser.add_argument("-Likability","--Likability", default=True,action="store_true")
    # parser.add_argument("-Coherence","--Coherence", default=True,action="store_true")
    # parser.add_argument("-Completeness","--Completeness", default=True,action="store_true")
    # parser.add_argument("-Vivacity","--Vivacity", default=True,action="store_true")
    
    # 위는 만들어진 결과물에 대한 evaluation.

    # 결과 자체에 대한 survey도 필요하지만,
    # 만드는 과정을 사용자가 직접 참여해볼 수 있는(특히, 씬 이미지를 대입하는 것도 포함해서) 싸이트도 필요할 것이다.
    # 이 경우에는 generate 함수를 그대로 가져가서, main function을 응용하여서 입력과 출력을 web으로 해줘야 한다.
    # 이 경우에는, 사용자가 직접 'Relavance' 수치 점수를 부여할 수 있다 (각각의 plot point 모두에.)
    # 물론, relavance 수치 자체가 좀 애매한 수치가 될 수 있기는 하다(우린 모델 자체에 relavance를 높이게 하기 위해 노력한 바가 따로 없음.)
    # relavance가 높다 해도 그것이 LLM 자체의 추론 능력인지 내 방법론 덕인지 알기가 좀 어렵다. 물론 잠재력을 최대한 끌어냈다고 말할 순 있겠지만.
    # 그보다는 '전반적인 플랫폼 만족도' 정도가 좋지 않을까 싶다.

    

    

    args = parser.parse_args()
    
    
    num_ver=0
    if os.path.exists(current_dir+"results/"+filename):
        for (path, d, files) in os.walk(current_dir+"results/"+filename):
            num_ver=len(d)
    
    datasets=[]
    
    for j in range(num_ver):
        lines=[]
        with jsonlines.open(os.path.join(os.path.dirname(current_dir), "results/"+args.text_file+"/generation_outputs_"+str(j)+".jsonl"), "r",) as f:
            for line in f:
                lines.append(line)
        datasets.append(lines)


    result={}

    if args.SBERT:
        result.update(SBERT(datasets))
    if args.ROUGE:
        result.update(ROUGE(datasets))
    if args.BLEU:
        result.update(BLEU(datasets))
    if args.in_Self_BLEU:
        result.update(in_Self_Bleu(datasets))
    if args.in_Self_SBERT:
        result.update(in_Self_SBERT(datasets))
    if args.Self_BLEU:
        result.update(Self_Bleu(datasets))
    if args.Self_SBERT:
        result.update(Self_SBERT(datasets))

    result['date'] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    result=[result]
    
    if os.path.exists(current_dir+"/results/"+ +args.text_file+"/evaluation_results.jsonl"):
        mode="a"
    else:
        mode="w"
    with jsonlines.open(current_dir + "/results/"+args.text_file+"/evaluation_results.jsonl", mode) as fw:
        fw.write_all(result)

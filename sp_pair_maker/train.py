import pipeline
import loader
import argparse

def trainer(train=True,use_GPT4=False):
    pl=pipeline.PipeLine(train=train,use_GPT4=use_GPT4)
    ld=loader.Loader()

    # train. valid, test.
    # we only make a arbitrary sentence-prompt pair for the train set.
    sentence_wp,_,_=ld.get_sentence_wp()
    

    for data in sentence_wp:
        new_story=True
        for pair in data["sent_prompt_pair"]:
            
            prompt=pl.get_prompt(pair["sentence"],new_story=new_story)
            image=pl.get_image(prompt)
            prob=pl.get_logits(image,prompt,pair["sentence"])
            pl.lm_update()
            new_story=False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train", action="store_true")
    parser.add_argument("-use_GPT4", "--use_GPT4", action="store_true")
    args = parser.parse_args()

    trainer(train=args.train,use_GPT4=args.use_GPT4)


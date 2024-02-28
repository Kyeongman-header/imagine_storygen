import os


def load_and_replace_words():
    data = ["train", "test", "valid"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "writingPrompts")
    for name in data:
        with open(folder_path+'/'+name + ".wp_target") as f:
            stories = f.readlines()
            stories = [i.replace("<newline>","") for i in stories]
            # trim은 하지 않고, 대신 성가신 <newline>을 지운다.
            # no trim, just replace the irritating word, <newline>.
            with open(folder_path+'/'+name + ".wp_target", "w") as o:
                for line in stories:
                    o.write(line.strip() + "\n")

        with open(folder_path+'/'+name + ".wp_source") as f:
            titles = f.readlines()
            titles = [i.replace("[ WP ]","") for i in titles]
            # replace <WP>.
            with open(folder_path+'/'+name + ".wp_source", "w") as o:
                for line in titles:
                    o.write(line.strip() + "\n")


if __name__ == "__main__":
    load_and_replace_words()


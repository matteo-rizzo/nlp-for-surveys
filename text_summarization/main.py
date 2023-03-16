import os.path
import time

from text_summarization.classes.SummarizersFactory import SummarizersFactory
from text_summarization.functional.settings import PATH_TO_LOG, PATH_TO_DATASET

"""
Supported methods: 'hugging_face', 'gpt3'
"""

METHOD = "gpt3"


def main():
    print("\n **********************************************")
    print(" Text Summarization Using '{}' Pipeline".format(METHOD))
    print(" ********************************************** \n")

    summarizer = SummarizersFactory().get(METHOD)

    path_to_log_dir = os.path.join(PATH_TO_LOG, "{}_{}".format(METHOD, time.time()))
    os.makedirs(path_to_log_dir)
    print("\n Logging at '{}' \n".format(path_to_log_dir))

    for file_name in os.listdir(PATH_TO_DATASET):
        if file_name == ".DS_store":
            continue

        path_to_file = os.path.join(PATH_TO_DATASET, file_name)
        print("\n Loading text file at {}...".format(path_to_file))

        text_file = open(path_to_file, "r")
        print("\n File loaded! \n")

        text = text_file.read().strip()
        print("\n Loaded text file contains {} characters. Text preview: \n {}".format(len(text), text))

        summary = summarizer.summarize(text)

        path_to_log_file = os.path.join(path_to_log_dir, file_name.split(".")[0] + ".txt")
        open(path_to_log_file, "w").write(summary)

        print("\n ********************************************** \n")


if __name__ == '__main__':
    main()

import os

import nltk

PATH_TO_DATASET = os.path.join("data", "nlp-for-surveys", "corpus")
PATH_TO_LOG = os.path.join("text_summarization", "logs")

nltk.download('punkt')

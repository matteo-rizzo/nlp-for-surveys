from text_summarization.classes.GPT3 import GPT3
from text_summarization.classes.HuggingFacePipeline import HuggingFacePipeline
from text_summarization.classes.Summarizer import Summarizer


class SummarizersFactory:

    def __init__(self):
        self.__summarizers = {
            "hugging_face": HuggingFacePipeline,
            "gpt3": GPT3
        }

    def get(self, summarizer: str) -> Summarizer:
        if summarizer not in self.__summarizers.keys():
            raise ValueError("Summarizer '{}' not implemented! Implemented summarizers are: {}"
                             .format(summarizer, self.__summarizers.keys()))
        return self.__summarizers[summarizer]()

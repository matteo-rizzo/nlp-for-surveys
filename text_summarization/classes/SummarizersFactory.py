from text_summarization.classes.BertExtractive import BertExtractive
from text_summarization.classes.GPT3 import GPT3
from text_summarization.classes.HuggingFaceAbstractive import HuggingFaceAbstractive
from text_summarization.classes.SummarizerPipeline import SummarizerPipeline


class SummarizersFactory:

    def __init__(self):
        self.__summarizers = {
            "hugging_face_abstractive": HuggingFaceAbstractive,
            "bert_extractive": BertExtractive,
            "gpt3": GPT3
        }

    def get(self, summarizer: str) -> SummarizerPipeline:
        if summarizer not in self.__summarizers.keys():
            raise ValueError("Summarizer '{}' not implemented! Implemented summarizers are: {}"
                             .format(summarizer, self.__summarizers.keys()))
        return self.__summarizers[summarizer]()

from typing import List

from summarizer.bert import TransformerSummarizer
from transformers import BertTokenizer

from text_summarization.classes.SummarizerPipeline import SummarizerPipeline


class BertExtractive(SummarizerPipeline):

    def __init__(self):
        super().__init__(tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))
        self.__model = TransformerSummarizer()

    def _make_summary(self, chunks: List) -> str:
        summary = []
        for chunk in chunks:
            chunk_summary = self.__model(chunk, num_sentences=1, min_length=60)
            summary.append(chunk_summary)
            print(chunk_summary)
        return "".join(summary)

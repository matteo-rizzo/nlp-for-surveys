from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from text_summarization.classes.Summarizer import Summarizer

CHECKPOINT = "sshleifer/distilbart-cnn-12-6"


class HuggingFacePipeline(Summarizer):

    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__(tokenizer=AutoTokenizer.from_pretrained(checkpoint))
        print("\n Loading model and tokenizer from checkpoint {}".format(checkpoint))
        self.__model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        print("\n Model and tokenizer loaded!")

    def _make_summary(self, chunks: List) -> str:
        summary = []
        for x in [self._tokenizer(chunk, return_tensors="pt") for chunk in chunks]:
            o = self.__model.generate(**x)
            chunk_summary = self._tokenizer.decode(*o, skip_special_tokens=True)
            summary.append(chunk_summary + "\n")
            print(chunk_summary)
        return "".join(summary)

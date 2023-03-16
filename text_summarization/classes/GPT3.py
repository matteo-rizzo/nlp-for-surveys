import os
from typing import List

import openai
from transformers import AutoTokenizer

from text_summarization.classes.Summarizer import Summarizer
from text_summarization.functional.secrets import OPENAI_API_KEY


class GPT3(Summarizer):

    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        super().__init__(tokenizer=AutoTokenizer.from_pretrained("openai-gpt"))
        openai.api_key = OPENAI_API_KEY

    @staticmethod
    def __api_call(chunk: str) -> str:
        openai_summary = openai.Completion.create(
            model="text-davinci-003",
            prompt="Summarize this:\n\n" + chunk,
            temperature=0.7,
            max_tokens=64,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return openai_summary.choices[0].text

    def _make_summary(self, chunks: List) -> str:
        summary = []
        for chunk in chunks:
            chunk_summary = self.__api_call(chunk)
            summary.append(chunk_summary)
            print(chunk_summary)
        return "".join(summary)

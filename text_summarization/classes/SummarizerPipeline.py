from abc import abstractmethod
from typing import List

import nltk
from transformers import PreTrainedTokenizerBase


class SummarizerPipeline:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self._tokenizer = tokenizer

    @staticmethod
    def _make_sentences(text: str) -> List:
        return nltk.tokenize.sent_tokenize(text)

    def _make_chunks(self, sentences: List) -> List:
        length, chunk, chunks, counter = 0, "", [], -1

        for sentence in sentences:
            counter += 1
            combined_length = len(self._tokenizer.tokenize(sentence)) + length

            if combined_length <= self._tokenizer.max_len_single_sentence:
                chunk += sentence + " "
                length = combined_length

                if counter == len(sentences) - 1:
                    chunks.append(chunk.strip())
            else:
                chunks.append(chunk.strip())

                length = 0
                chunk = ""

                chunk += sentence + " "
                length = len(self._tokenizer.tokenize(sentence))

        return chunks

    @abstractmethod
    def _make_summary(self, chunks: List) -> str:
        pass

    def summarize(self, text: str) -> str:
        sentences = self._make_sentences(text)
        print("\n Text split in {} SENTENCES. The max tokens in longest sentence is {} \n"
              .format(len(sentences), max([len(self._tokenizer.tokenize(sentence)) for sentence in sentences])))

        chunks = self._make_chunks(sentences)
        print("\n Text split in {} CHUNKS: \n"
              "\t Chunks len wo/ special tokens: .......... {} \n"
              "\t Chunks len w/ special tokens: ........... {} \n"
              "\t Sum of chunks len wo/ special tokens: ... {}/{} \n"
              "\t Sum of chunks len w/ special tokens: .... {}/{} \n"
              .format(len(chunks), [len(self._tokenizer.tokenize(c)) for c in chunks],
                      [len(self._tokenizer(c).input_ids) for c in chunks],
                      sum([len(self._tokenizer.tokenize(c)) for c in chunks]), len(self._tokenizer.tokenize(text)),
                      sum([len(self._tokenizer(c).input_ids) for c in chunks]), len(self._tokenizer(text).input_ids)))

        print("\n SUMMARY:")
        print(" ............................................................ ")
        summary = self._make_summary(chunks)
        print("\n ............................................................ \n")

        return summary

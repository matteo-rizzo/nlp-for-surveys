import re
from string import punctuation

import spacy
from nltk import word_tokenize
from spacy import Language
# from spacy.lang.en import STOP_WORDS
from spacy.tokens import Span, Doc
import pytextrank

from topic_extraction.BaseTopicExtractor import TopicExtractor


@spacy.registry.misc("articles_scrubber")
def articles_scrubber():
    def scrubber_func(span: Span) -> str:
        if span[0].ent_type_:
            # ignore named entities
            return "INELIGIBLE_PHRASE"
        for token in span:
            if token.pos_ not in ["DET", "PRON"]:
                break
            span = span[1:]
        return span.text

    return scrubber_func


def get_topics_from_document(doc: Doc, k: int) -> list[tuple[str, float]]:
    # examine the top-ranked phrases in the document
    topics = list()
    for phrase in doc._.phrases:
        if phrase.text != "INELIGIBLE_PHRASE":
            topics.append((phrase.text, phrase.rank))
        if len(topics) == k:
            break
    return topics


@Language.component("custom_cleaning")
def custom_cleaning(doc: Doc) -> Doc:
    sentence: str = doc.text
    # sentence = sentence.lower()
    # sentence = sentence.replace('{html}', "")
    # cleanr = re.compile('<.*?>')
    # cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', sentence)
    rem_strange = re.sub(r'<[()]>@\\/', '', rem_url)
    rem_num = re.sub('[0-9]+', '', rem_strange)
    rem_spaces = re.sub('\s+', ' ', rem_num).strip()
    # tokenizer = RegexpTokenizer(r'\w+')
    tokens = word_tokenize(rem_spaces)
    # tokens = rem_spaces.split(sep=" ")
    # filtered_words = [w for w in tokens if len(w) > 2 if not w in STOP_WORDS]
    has_spaces = [w2 not in punctuation for w1, w2 in zip(tokens, tokens[1:])]
    has_spaces.append(False)  # last word has no spaces

    new_doc = Doc(doc.vocab, words=tokens, spaces=has_spaces)

    return new_doc


class TopicRank(TopicExtractor):
    def __init__(self):
        self._nlp = None

    def prepare(self):
        self._nlp = spacy.load("en_core_web_md")

        method = ["topicrank", "textrank"][1]
        # add PyTextRank to the spaCy pipeline
        self._nlp.add_pipe(method, config={"stopwords": {"word": ["NOUN"]}, "scrubber": {"@misc": "articles_scrubber"}})
        self._nlp.add_pipe("custom_cleaning", first=True)

    def extract(self, document: str, k: int, *args, **kwargs) -> list[tuple[str, float]]:
        doc = self._nlp(document)
        return get_topics_from_document(doc, k)

    def batch_extract(self, documents: list[str], k: int, *args, **kwargs) -> list[list[tuple[str, float]]]:
        docs = [self._nlp(text) for text in documents]
        all_docs_topics = list()
        for doc in docs:
            topics = get_topics_from_document(doc, k)
            all_docs_topics.append(topics)
        return all_docs_topics

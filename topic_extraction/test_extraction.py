from topic_extraction import BaseTopicExtractor
from topic_extraction.TextRank import TopicRank
from topic_extraction.extract_metadata import text_extraction


# example text
# text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."


def extraction_test(model: BaseTopicExtractor):
    articles, titles = text_extraction()
    topics: list[list[tuple[str, float]]] = model.batch_extract(articles, k=5)
    with open("test.txt", mode="w") as f:
        for title, tops in zip(titles, topics):
            f.write(f"{title} - [{','.join([t[0] for t in tops])}]\n")


if __name__ == "__main__":
    algo = TopicRank()
    algo.prepare()
    extraction_test(algo)

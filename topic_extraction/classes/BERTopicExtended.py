from __future__ import annotations

import pandas as pd
from bertopic import BERTopic
from bertopic._utils import check_is_fitted
import plotly.graph_objects as go

from topic_extraction.visualization.plotly_utils import visualize_topics_over_time_ext


class BERTopicExtended(BERTopic):

    def visualize_topics_over_time(self,
                                   topics_over_time: pd.DataFrame,
                                   top_n_topics: int = None,
                                   topics: list[int] = None,
                                   normalize_frequency: bool = False,
                                   relative_frequency: bool = False,
                                   custom_labels: bool = False,
                                   title: str = "<b>Topics over Time</b>",
                                   width: int = 1250,
                                   height: int = 450) -> go.Figure:
        """ Visualize topics over time

        Arguments:
            topics_over_time: The topics you would like to be visualized with the
                              corresponding topic representation
            top_n_topics: To visualize the most frequent topics instead of all
            topics: Select which topics you would like to be visualized
            normalize_frequency: Whether to normalize each topic's frequency individually
            relative_frequency: Whether to show the relative frequency. Overrides normalize_frequency
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            A plotly.graph_objects.Figure including all traces

        Examples:

        To visualize the topics over time, simply run:

        ```python
        topics_over_time = topic_model.topics_over_time(docs, timestamps)
        topic_model.visualize_topics_over_time(topics_over_time)
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_topics_over_time(topics_over_time)
        fig.write_html("path/to/file.html")
        ```
        """
        check_is_fitted(self)
        return visualize_topics_over_time_ext(self,
                                              topics_over_time=topics_over_time,
                                              top_n_topics=top_n_topics,
                                              topics=topics,
                                              normalize_frequency=normalize_frequency,
                                              relative_frequency=relative_frequency,
                                              custom_labels=custom_labels,
                                              title=title,
                                              width=width,
                                              height=height)

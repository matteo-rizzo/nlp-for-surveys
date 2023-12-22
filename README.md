# nlp-for-surveys

NLP pipeline of topic extraction, text summarization, and useful visualizations for surveys

## Descriptions

- **Topic stacked**: document embeddings are reduced to 2-dimensional space for visualization purposes. We used word
  embeddings from a language model pre-trained on scientific publications. This visualization allows getting an overview
  of how documents belonging to each subject are spread across themes.
  Documents belonging to the "digitalization" cluster are marked with a circle, while documents assigned to
  the "sustainability" cluster are marked with a cross.
- **Theme per subject**: this plot helps to understand how themes are distributed in subject clusters. In other words,
  it shows how much overlap is present between subjects and the two themes, measured in terms of the percentage of
  documents for each subject.
- **Topic evolution (in time)**: shows how the representative keywords of each topic (subject) evolved over time. To do
  this, the top-k representative words of each subject are re-computed for each year, only considering the documents
  published in that year. In the plot, the frequency of documents is normalized separately for each subject (meaning
  that the values in the plot for each subject sum up to 1).
 
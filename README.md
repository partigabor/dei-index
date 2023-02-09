# dei-index

The following code (`main.ipynb`) is a pipeline of NLP solution(s) to quickly analyze and compare company documents. Currently the program can read company reports (`pdf`) and output a set of observations about said reports' contents, quantifying relative mentions of certain key terms and phrases, and provide simple visualizations. It uses [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) scores (term frequency–inverse document frequency), a common technique known from text mining and information retrieval. This metric, TF-IDF takes the frequency of a term in a documents, multiplied by the log of the term's inverse document frequency (the number of documents it appears divided by the total number of documents), resulting in higher scores if a term is unique, and lower scores if a term is common across the corpus. 

This brief example is focusing on the terms 'diversity', 'equity', and 'inclusion' in CSR reports of two big beverage companies over the past years. For a big data approach, I recommend using the [Jena Organization Corpus (JOCo)](https://www.orga.uni-jena.de/en/corp) which is a 280 million word corpus of US, UK, and German company reports.

>The ultimate goal is the creation of an index to capture and measure companies DEI practices and initiatives. 

At present, this program can:
* read in and pre-process txt and pdf files of company documents and reports, 
* collate their contents in a dataframe
* tokenize, remove stopwords, and lemmatize text
* calculate tf-idf scores for every document in the corpus
* compare a set of selected documents and visualize the comparison

>This code was tested on a local machine, on Windows, using VSCode and Python 3.9.13 via Anaconda.

Gabor Parti, 2022 October

If you have any questions, contact me at gabor.parti@connect.polyu.hk

<!-- ## Examples

All three examples below show a segment of observations from a 200 million-word corpus of company reports, using TF-IDF (term frequency–inverse document frequency) scores:

### Example 1
* [Company reports of 2011 and their mentions of a few key phrases using TF-IDF scores](https://htmlpreview.github.io/?https://github.com/partigabor/dei-index/blob/master/examples/2011.html)

### Example 2
* [3 companies and their focus on 'diversity' and 'inclusion'](https://htmlpreview.github.io/?https://github.com/partigabor/dei-index/blob/master/examples/dei.html)

### Example 3
* [The two cola companies, and the distribution of key terms' tf-idf scores in their reports over the years](https://htmlpreview.github.io/?https://github.com/partigabor/dei-index/blob/master/examples/cola.html) -->

# %% [markdown]
# # DEI Index
# 
# The following code (`main.ipynb`) is a pipeline of NLP solution(s) to quickly analyze and compare company documents. Currently the program can read company reports and output a set of observations about said reports' contents, quantifying relative mentions of certain key terms and phrases, and provide simple visualizations. It uses [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) scores (term frequency–inverse document frequency), a common technique known from text mining and information retrieval. This metric TF-IDF takes the frequency of a term in a documents, multiplied by the log of the term's inverse document frequency (the number of documents it appears divided by the total number of documents), resulting in higher scores if a term is unique, and low scores if a term is common across the corpus. This brief example is focusing on the terms 'diversity', 'equity', and 'inclusion' in CSR reports of two big beverage companies over the past years. For a big data approach, I recommend using the [Jena Organization Corpus (JOCo)](https://www.orga.uni-jena.de/en/corp) which is a 280 million word corpus of US, UK, and German company reports.
# 
# >The ultimate goal is the creation of an index to capture and measure companies DEI practices and initiatives. 
# 
# At present, this program can:
# * read in and pre-process txt and pdf files of company documents and reports, 
# * collate their contents in a dataframe
# * tokenize, remove stopwords, and lemmatize text
# * calculate tf-idf scores for every document in the corpus
# * compare a set of selected documents and visualize the comparison
# 
# >This code was tested on a local machine, on Windows, using VSCode and Python 3.9.13 via Anaconda, but you could try it on Google Colab.
# 
# Gabor Parti, 2022 October
# If you have any questions, contact me at gabor.parti@connect.polyu.hk

# %% [markdown]
# ## Setup

# %%
# # mount your Google drive if you use Google Colaboratory.
# from google.colab import drive
# drive.mount('/content/drive')

# Warning! Paths are a bit messed up if using colab so regular expressions may not work the same as locally.

# %%
# install necessary dependencies
# %pip install PyPDF2

# %%
#import dependency libraries
import pandas as pd
import numpy as np
import regex as re
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
import sys
import os
import PyPDF2
import sklearn as sk
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

# %% [markdown]
# ## Preprocessing Reports
# 
# The following two functions will be used to read in and parse documents from `pdf` and `txt` files, clean their contents (remove symbols and punctuation, lowercase) and store them.
# 
# ### Functions

# %%
# function to read a pdf file and add them to a dataframe
def read_pdf(document, index=0):
  """Read and parse a pdf file.
  This function uses the PyPDF2 package to read and extract 
  the contents of a pdf file, page by page.

  Keyword arguments:
  document -- the document to be read in.
  index -- the index of the document, an integer (default 0)
  """
  # add filename to dataframe
  m = re.search(r"\\(?!.*\\)(.*)_(\d+)(_?.*)?((\.pdf)|(\.txt))", document.lower())
  filename_match = m.group(0)
  filename = re.sub("[\\\/\'\>]", "", filename_match)
  filename = re.sub("\.\w+", "", filename)
  company = m.group(1)
  year = m.group(2)
  type = m.group(3)
  type = re.sub("^_", "", type)
  ext = m.group(4)
  ext = re.sub("\.", "", ext)

  df.loc[index, 'file'] = filename
  df.loc[index, 'company'] = company
  df.loc[index, 'year'] = year
  df.loc[index, 'type'] = type
  if type == "":
    df.loc[index, 'type'] = 'not_csr'
  
  print("Parsing", filename, "...")

  # #####
  # y = re.search("([0-9]{4})", filename)
  # if y is None:
  #   df.loc[index, 'year'] = np.nan
  # else:
  #   df.loc[index, 'year'] = y[0]
  # #####

  # creating a pdf file object
  pdfFileObj = open(document, 'rb') 

  # n = re.search(r"\\(?:.(?!\\))+$", filename_with_path)
  # filename_match = n.group(0)

  # creating a pdf reader object 
  pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict=False) 
      
  # printing number of pages in pdf file 
  # print("Number of pages:", pdfReader.numPages)
  pages = pdfReader.numPages

  # add page number to dataframe
  # df.loc[index, 'pages'] = pages

  # creating a page object 
  pageObj = pdfReader.getPage(0)

  # extracting text from page 
  # print(pageObj.extractText())

  pages_with_contents = []

  for p in range(pages):
    pageObj = pdfReader.getPage(p)
    page_contents = pageObj.extractText()
    pages_with_contents.append(page_contents)

  # join pages into one document
  contents = " ".join(pages_with_contents)

  #closing the pdf file object 
  pdfFileObj.close() 

  # cleaning
  contents = re.sub("\n", " ", contents)
  contents = re.sub("\.", ". ", contents)
  contents = re.sub("\)", ") ", contents)

  # separates words
  contents = re.sub(r"([a-z])([A-Z])", r"\1 \2", contents)

  # # if need list
  # contents_list = contents.split(' || ')

  # remove symbols
  contents = re.sub(r"[^a-zA-Z0-9]", " ", contents)

  # # lowercase
  contents = contents.lower()

  #remove extra spaces
  contents = re.sub("\s+", " ", contents)

  #from where
  df.loc[index, 'source'] = "manual"

  # add contents to dataframe
  df.loc[index, 'contents'] = contents

  return #print("Done.")

# %%
# function to read a txt file and add them to a dataframe
def read_txt(document, index=0):
  """Read and parse a txt file.

  Keyword arguments:
  document -- the document to be read in.
  index -- the index of the document, an integer (default 0)
  """
  # creating a pdf file object 
  with open(document, encoding='utf-8') as f:
    contents = f.read()

  # add filename to dataframe
  m = re.search(r"\\(?!.*\\)(.*)_(\d+)(_?.*)?((\.pdf)|(\.txt))", document.lower())
  filename_match = m.group(0)
  filename = re.sub("[\\\/\'\>]", "", filename_match)
  filename = re.sub("\.\w+", "", filename)
  company = m.group(1)
  year = m.group(2)
  type = m.group(3)
  type = re.sub("^_", "", type)
  ext = m.group(4)
  ext = re.sub("\.", "", ext)

  df.loc[index, 'file'] = filename
  df.loc[index, 'company'] = company
  df.loc[index, 'year'] = year
  df.loc[index, 'type'] = type
  if type == "":
    df.loc[index, 'type'] = 'not_csr'

  print("Parsing", filename, "...")

  y = re.search("([0-9]{4})", filename)
  if y is None:
    df.loc[index, 'year'] = np.nan
  else:
    df.loc[index, 'year'] = y[0]

  f.close()

  # cleaning
  contents = re.sub("\n", " ", contents)
  contents = re.sub("\.", ". ", contents)
  contents = re.sub("\)", ") ", contents)

  # remove symbols
  contents = re.sub(r"[^a-zA-Z0-9]", " ", contents)

  # lowercase
  contents = contents.lower()

  #remove extra spaces
  contents = re.sub("\s+", " ", contents)

  # from where
  df.loc[index, 'source'] = "joco"

  # add contents to dataframe
  df.loc[index, 'contents'] = contents

  return #print("Done.")

# %%
# a function to walk through all files in a folder and its subfolders
def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(os.path.join(subdir, file))                                                                         
    return r

# %% [markdown]
# ### Preprocessor
# 
# The following block of code reads in a directory (a folder) where company reports should be placed. It iterates over the files in that directory, and if they are in the correct file extension (`pdf`, `txt`), then the program will parse the files. Including: tokenization, counting tokens (~words), removing stopwords, lemmatization, and placing them all in one dataframe, where every row represents a document. The input folder should contain files with the following filename conventions: "company_year_type.txt" or "company_year_type.pdf", where 'type' can be CSR or ESG or something else, and 'year' must be digits. 
# 
# E.g.: "CocaCola_2021_CSR.pdf" or "Pepsico_2011.txt" are both valid filenames. Capitalization does not matter.

# %%
# assign relative directory
directory = os.path.join(sys.path[0], "testdata\\csr") ### INPUT FOLDER HERE ### "testdata\\csr"
print("Your input directory is:", directory)

# list files in directory
files_in_dir = list_files(directory)
# files_in_dir = os.listdir(directory)

# count files in directory
print("Number of files:",len(files_in_dir))

# initialize dataframe to hold documents
df = pd.DataFrame(columns=['file'])

# iterate over files in the directory
misc_files = []
i = 0
for f in files_in_dir:
    if f.lower().endswith('.pdf'):
        # print("Found pdf,")
        read_pdf(f,i)
    elif f.lower().endswith('.txt'):
        # print("Found txt,")
        read_txt(f,i)
    # elif f.lower().endswith('desktop.ini'):
    #     print("Hmm...")
    else:
        print("Found something else.")
        misc_files.append(f)
    i = i + 1

if len(misc_files) > 0:
    print("Warning, some files with dubious extensions were found but not parsed:", print(misc_files))
else:
    print("All files read in.")

# tokenize contents
print("Tokenizing data...")
df['tokenized'] = df.contents.copy().apply(lambda x: nltk.word_tokenize(x))

# count words/tokens
print("Counting words...")
df['tokens'] = df.tokenized.copy().apply(lambda x: len(x))

# drop empty rows
print(df.shape[0], "documents, dropping empty ones if any...")
df = df[df['tokens'] > 0]
print(df.shape[0], "remaining.")

# set stopwords from nltk
stop = set(stopwords.words('english'))

# remove stopwords
print("Removing stopwords...")
df['without_stopwords'] = df['tokenized'].copy().apply(lambda x: ' '.join([word for word in x if word not in (stop)]))

# define lemmatizer module from nltk
lemmatizer = WordNetLemmatizer()

# lemmatize
print("Lemmatizing data...")
df['lemmatized'] = df['without_stopwords'].copy().apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# reorder columns
# df = df[['file', 'source', 'year', 'tokens', 'contents', 'tokenized', 'without_stopwords', 'lemmatized',]]

# # drop unnecessary rows
df = df.drop(columns=['tokenized','without_stopwords'])

# drop addenda reports (optional)
no_of_addenda = len(df[df['file'].str.contains("add")])
print("Found", no_of_addenda, "addenda, dropping it.")
df = df[df['file'].str.contains("add") == False]
df.reset_index(inplace=True, drop=True)

# export
print("Exporting...")
df.to_csv("parsed_documents.csv")

# done
print("All done, no errors.")
df

# %% [markdown]
# ## TF-IDF
# 
# The following block of code will read the dataframe of the parsed documents (imported from a previously saved `csv`), and we define the dataset to capture **tf-idf** (term frequency–inverse document frequency) scores of terms (unigrams, bigrams, trigrams, etc.) throughout the documents. Then, we should give list of the target terms we are looking for in the set of documents, in this example I am going to focus on ['diversity', 'equity', 'inclusion']. 
# 
# >Note: If you use the lemmatized contents, don't forget to search for singular terms instead of plural (e.g. 'human right' instead of 'human rights').
# 
# You can also add additional "stopwords", words you want the vectorizer to ignore. 
# We calculate the tf-idf scores for unigrams by iterating through every document and join the outputs together to get a dataframe that contains the scores of every document for the specific terms we are looking for. 
# 
# >Note that you can change the ngram range in line 14!
# 
# Here we could also look at the top most "important/salient" terms in a specific document, relative to all the other documents in the corpus.

# %% [markdown]
# ### Solution 1
# 
# Obtain a simple set of scores on target terms that are easy to plot and manipulate.

# %%
# load in preprocessed data
df = pd.read_csv("parsed_documents.csv", index_col=0)

# define the dataset as a list of document contents (text)
dataset = df['lemmatized'].tolist()

# define a list of target terms (keywords and phrases) to filter for later
filter = ['diversity', 'equity', 'inclusion']

# manually add to the list of stopwords if needed
custom_stop_words = text.ENGLISH_STOP_WORDS.union([""])

# set vectorizer
tfIdfVectorizer=TfidfVectorizer(use_idf=True, ngram_range=(1,1), stop_words=custom_stop_words)

# turn text into tf-idf vectors
tfIdf = tfIdfVectorizer.fit_transform(dataset)

# correlation matrix (for later)
corr_matrix = ((tfIdf * tfIdf.T).A)

# list of filenames and their no.
filenames = df['file'].tolist()
filenames = [re.sub("\.\w+", '', i) for i in filenames]
no_of_files = len(filenames)

# initialize dataframe to hold tfidf scores
df_all_scores = pd.DataFrame(columns=['term'])
df_filtered_scores = pd.DataFrame(columns=['term'])

# loop through all documents and get scores for each term term
for i in range(no_of_files):
    # get tf-idf scores for words/phrases
    print("Working on #" + str(i) + ",", filenames[i])
    df_scores = pd.DataFrame(tfIdf[i].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=[filenames[i]])

    # sort values according to score
    df_scores = df_scores.sort_values(filenames[i], ascending=False)

    # reset index and rename it term
    df_scores.reset_index(inplace=True)
    df_scores.rename(columns = {'index':'term'}, inplace = True)

    # merge all tfidf scores
    # df_all_scores = pd.merge(df_all_scores, df_scores.head(100), how='outer', on = 'term')
    # df.fillna(0.0)

    # filter for manual selection 
    df_filtered = df_scores[df_scores['term'].isin(filter)]

    # merge new tfidf scores with the rest
    df_filtered_scores = pd.merge(df_filtered_scores, df_filtered, how='outer', on = 'term')

# export
df_filtered_scores.to_csv("tfidf.csv")

df_filtered_scores

# %% [markdown]
# #### Plot
# 
# The following cell creates a plot using the [plotly](https://plotly.com/python/) Python visualization library. Before the actual plot, you can customize what you want to see.

# %%
# read in data
df_plot = pd.read_csv("tfidf.csv", index_col=0)

df_plot = df_plot.set_index('term')

# further narrow within filtered terms (optional)
# df_plot = df_plot[(df_plot['term'] == 'diversity') | (df_plot['term'] == 'inclusion')]

# filter columns to be plotted by searching the column name (e.g. year)
# cols1 = [col for col in df_plot.columns if 'hsbc' in col]
# cols2 = [col for col in df_plot.columns if 'pepsi' in col]
# df_plot = df_plot[cols1]

# transpose dataframe
df_plot = df_plot.transpose()

# try to capture other details from filename (ignore now)
# company = []    
# for values in df_plot['term']:
#     company.append(re.search(r"(.*)_(\d+)_?(.*)?", values).group(1))
# df_plot['company'] = company

# ######

# plot it
fig = px.bar(df_plot)#, facet_col="term")
fig.show()

# export it
filename="test_plot_1"
fig.write_html(filename + ".html")

# %% [markdown]
# ### Heatmap of correlation matrix 
# Using the tf-idf metrics

# %%
# get heatmap to notice "zones" and outliers easier
fig = px.imshow(corr_matrix)
fig.show()

# %% [markdown]
# ### Solution 2 (Currently cumbersome)
# Appends the selected terms' scores to the original dataset which accommodates more possibilities for plotting, but makes plotting settings more tedious. 
# 
# (Pros: can use other parameters such as 'year' and 'company'; 
# Cons: cannot filter for 'term')

# %%
# load in preprocessed data
df = pd.read_csv("parsed_documents.csv", index_col=0)

# define the dataset as a list of document contents (text)
dataset = df['lemmatized'].tolist()

# define a list of target terms (keywords and phrases) to filter for later
filter = ['diversity', 'equity', 'inclusion']

# manually add to the list of stopwords if needed
custom_stop_words = text.ENGLISH_STOP_WORDS.union([""])

# set vectorizer
tfIdfVectorizer=TfidfVectorizer(use_idf=True, ngram_range=(1,1), stop_words=custom_stop_words)

# turn text into tf-idf vectors
tfIdf = tfIdfVectorizer.fit_transform(dataset)

# correlation matrix (for later)
corr_matrix = ((tfIdf * tfIdf.T).A)

# list of filenames and their no.
filenames = df['file'].tolist()
filenames = [re.sub("\.\w+", '', i) for i in filenames]
no_of_files = len(filenames)

# initialize dataframe to hold tfidf scores
# df_all_scores = pd.DataFrame(columns=[])
df_filtered_scores = pd.DataFrame(columns=['file'])
df_merged_tfidf_scores = pd.DataFrame(columns=['file'])

# loop through all documents and get scores for each term term
for i in range(no_of_files):
    # get tf-idf scores for words/phrases
    print("Working on #" + str(i) + ",", filenames[i])
    df_tfidf = pd.DataFrame(tfIdf[i].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=[filenames[i]])

    # sort values according to score
    df_tfidf = df_tfidf.sort_values(filenames[i], ascending=False)

    # merge all tfidf scores
    # df_all_scores = pd.merge(df_all_scores, df_tfidf.head(100), how='outer', on = 'term')
    # df_all_scores.fillna(0.0)

    # filter for manual selection 
    df_filtered_tfidf = df_tfidf[df_tfidf.index.isin(filter)]

    # transpose dataframe
    df_filtered_tfidf = df_filtered_tfidf.transpose()

    # reset index and rename it term
    df_filtered_tfidf.reset_index(inplace=True)
    df_filtered_tfidf.rename(columns = {'index':'file'}, inplace = True)

    # merge new tfidf scores with the rest
    df_merged_tfidf_scores=pd.concat([df_merged_tfidf_scores, df_filtered_tfidf])

# merge tfidf scores with the dataset
df = pd.merge(df, df_merged_tfidf_scores, on = 'file')

# export
df.to_csv("parsed_documents_with_tfidf.csv")
# df_all_scores.to_csv("all_scores.csv")

# %%
df_plot

# %% [markdown]
# #### Plot
# 
# Next we plot the results using plotly.

# %%
# # read in data
df_plot = pd.read_csv("parsed_documents_with_tfidf.csv", index_col=0)

# drop unnecessary rows
df_plot = df_plot.drop(columns=['source','contents','lemmatized'])

# filter rows to be plotted by searching the column name (e.g. year)
# df_plot = df_plot[(df_plot['file'].str.contains("bank")) | (df_plot['file'].str.contains("hsbc"))]
# cols1 = [col for col in df_plot.columns if '2007' in col]
# cols2 = [col for col in df_plot.columns if 'pepsi' in col]
# df_plot = df_plot[cols1]

# plot it
fig = px.bar(df_plot, x='year', y='diversity', color='company',
    facet_col="type"
    )
fig.show()

# export it
filename="test_plot_2"
fig.write_html(filename + ".html")

# %% [markdown]
# # Topic Modeling

# %%
# Install dependencies
# !pip install scattertext
# !pip install empath
# !pip install spacy
# !python -m spacy download en_core_web_sm ### CHOOSE ONE ###

# %%
# Import dependencies
import scattertext as st
# import spacy

# %%
category = 'type'
A = 'csr'
B = 'not_csr'

feat_builder = st.FeatsFromOnlyEmpath()
empath_corpus = st.CorpusFromParsedDocuments(df,
                                              category_col=category,
                                              feats_from_spacy_doc=feat_builder,
                                              parsed_col='contents').build()
                                              
html = st.produce_scattertext_explorer(empath_corpus,
                                       category=A,
                                       category_name=A,
                                       not_category_name=B,
                                       width_in_pixels=1000,
                                       metadata=df['file'],
                                       use_non_text_features=True,
                                       use_full_doc=True,
                                       topic_model_term_lists=feat_builder.get_top_model_term_lists())

open("test_tm.html", 'wb').write(html.encode('utf-8'))

# %% [markdown]
# # End

# %% [markdown]
# ## Notes



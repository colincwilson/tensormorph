Preprocessed data should be a pickled Python dictionary with the following entries (others are ignored):

- **dat**, **dat_train**, **dat_test**, **dat_val**  
  Pandas DataFrames containing all of the examples / train split / test split / val split.  

- **segments**  
  List of ordinary symbols (not begin/end delimiters, epsilon, or other specials) that are present in the data, each a unicode character string (without whitespace).

- **vowels**  
  List of segments classified as vowels.

- **max_len**  
  Maximum length of forms that can be represented (i.e., number of tensor-product roles).

Each Pandas DataFrame containing the full data or a split should have the following fields (others are ignored):

- **stem**  and **output**  
  Space-delimited sequences of ordinary symbols (no delimiters, padding, etc.).

- **morphosyn**  
  todo: document
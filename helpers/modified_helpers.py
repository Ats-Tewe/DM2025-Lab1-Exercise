import nltk

def format_rows(df, text_column='text'):
    D = []
    for text in df[text_column]:
        temp_d = " ".join(str(text).split("\n")).strip('\n\t ')
        D.append(temp_d)   # ← fixed here (no [temp_d])
    return D


def format_labels(df, label_column='label'):
    """Return numeric labels directly from DataFrame."""
    return df[label_column]

def check_missing_values(df):
    """
    Checks and returns how many missing (NaN) values are present
    in each column of the DataFrame.
    """
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()
    # return f"Total missing values: {total_missing}\n{missing_summary}"
    return f"Total missing values: {total_missing}. Summary: {missing_summary}"



def tokenize_text(text, remove_stopwords=False):
    """
    Tokenizes text into sentences and words using NLTK.
    """
    tokens = []
    for sent in nltk.sent_tokenize(str(text), language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            tokens.append(word)
    return tokens


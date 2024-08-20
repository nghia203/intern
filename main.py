import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest

def preprocess_text(text):
    """Tokenizing, removing stop words, stemming"""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

def summarize_meeting(transcript, num_sentences=5):
    """TF-IDF"""
    sentences = sent_tokenize(transcript)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    # Calculate sentence scores based on TF-IDF
    sentence_scores = [sum(tfidf_matrix[i, j] for j in range(tfidf_matrix.shape[1])) / len(preprocessed_sentences[i]) 
                       for i in range(len(sentences))]

    # Get top-scoring sentences
    top_sentence_indices = nlargest(num_sentences, range(len(sentence_scores)), key=lambda i: sentence_scores[i])

    summary = ' '.join([sentences[i] for i in top_sentence_indices])
    return summary

# Get input and output file paths
input_file = input("Enter the path to the input transcript file: ")
output_file = input("Enter the path to the output summary file: ")

# Read the transcript
with open(input_file, 'r') as f:
    transcript = f.read()

# Summarize the meeting
summary = summarize_meeting(transcript)

# Write the summary
with open(output_file, 'w') as f:
    f.write(summary)

print(f"Summary written to {output_file}")
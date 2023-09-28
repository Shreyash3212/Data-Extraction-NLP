import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from textblob import TextBlob
import syllables
from textstat import flesch_kincaid_grade

nltk.download('punkt')

# Load input data
input_file = "Input.xlsx"
output_file = "Output.xlsx"  # New output file
df_input = pd.read_excel(input_file)

# Create a copy of the input DataFrame for the output
df_output = df_input.copy()

# Initialize lists to store analysis results
positive_scores = []
negative_scores = []
polarity_scores = []
subjectivity_scores = []
avg_sentence_lengths = []
percentage_complex_words = []
fog_indexes = []
avg_words_per_sentence = []
complex_word_counts = []
word_counts = []
syllables_per_words = []
personal_pronouns = []
avg_word_lengths = []

# Function to extract article title and text from a URL
def extract_title_and_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Define selectors or tags that typically contain article titles and text
        title_selector = "h1"  # Example: <h1 class="article-title">Title Here</h1>
        text_selector = "p"    # Example: <p class="article-paragraph">Paragraph text here</p>
        
        # Extract title and article text using specified selectors
        title = soup.select_one(title_selector)
        text_elements = soup.select(text_selector)
        
        if title:
            title = title.get_text().strip()
        
        if text_elements:
            article_text = " ".join([element.get_text().strip() for element in text_elements])
        else:
            article_text = ""
        
        return title, article_text
    except Exception as e:
        print(f"Error extracting from {url}: {str(e)}")
        return None, None

# Function to calculate text analysis variables
def analyze_text(text):
    # Calculate TextBlob sentiment scores
    blob = TextBlob(text)
    positive_scores.append(blob.sentiment.polarity)
    negative_scores.append(blob.sentiment.subjectivity)
    polarity_scores.append(blob.sentiment.polarity)
    subjectivity_scores.append(blob.sentiment.subjectivity)
    
    # Calculate Flesch-Kincaid Grade Level
    grade_level = flesch_kincaid_grade(text)
    fog_indexes.append(grade_level)
    
    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Create a Pyphen object for syllable counting
    dic = syllables.estimate
    
    # Calculate average sentence length
    sentences = nltk.sent_tokenize(text)
    avg_sentence_lengths.append(len(words) / len(sentences))
    
    # Calculate percentage of complex words (words with more than two syllables)
    syllable_count = sum([dic(word) for word in words])
    percentage_complex_words.append((syllable_count / len(words)) * 100)
    
    # Calculate average number of words per sentence
    avg_words_per_sentence.append(len(words) / len(sentences))
    
    # Count complex words (words with more than two syllables)
    complex_word_count = sum([1 for word in words if dic(word) > 2])
    complex_word_counts.append(complex_word_count)
    
    # Total word count
    word_counts.append(len(words))
    
    # Calculate syllables per word
    syllables_per_words.append(syllable_count / len(words))
    
    # Count personal pronouns (e.g., I, you, he, she, etc.)
    personal_pronoun_count = sum([1 for word in words if word.lower() in ['i', 'you', 'he', 'she', 'it', 'we', 'they']])
    personal_pronouns.append(personal_pronoun_count)
    
    # Calculate average word length
    avg_word_length = sum([len(word) for word in words]) / len(words)
    avg_word_lengths.append(avg_word_length)

# Loop through each URL in the input file and extract title and article text
for index, row in df_output.iterrows():
    url = row['URL']
    title, article_text = extract_title_and_text(url)
    
    if title:
        df_output.at[index, 'Title'] = title  # Add 'Title' column to the output DataFrame
    
    if article_text:
        analyze_text(article_text)
        
        # Update the output DataFrame with the analysis results
        df_output.at[index, 'POSITIVE SCORE'] = positive_scores[-1]
        df_output.at[index, 'NEGATIVE SCORE'] = negative_scores[-1]
        df_output.at[index, 'POLARITY SCORE'] = polarity_scores[-1]
        df_output.at[index, 'SUBJECTIVITY SCORE'] = subjectivity_scores[-1]
        df_output.at[index, 'AVG SENTENCE LENGTH'] = avg_sentence_lengths[-1]
        df_output.at[index, 'PERCENTAGE OF COMPLEX WORDS'] = percentage_complex_words[-1]
        df_output.at[index, 'FOG INDEX'] = fog_indexes[-1]
        df_output.at[index, 'AVG NUMBER OF WORDS PER SENTENCE'] = avg_words_per_sentence[-1]
        df_output.at[index, 'COMPLEX WORD COUNT'] = complex_word_counts[-1]
        df_output.at[index, 'WORD COUNT'] = word_counts[-1]
        df_output.at[index, 'SYLLABLE PER WORD'] = syllables_per_words[-1]
        df_output.at[index, 'PERSONAL PRONOUNS'] = personal_pronouns[-1]
        df_output.at[index, 'AVG WORD LENGTH'] = avg_word_lengths[-1]

# Save the analysis results to the output file
df_output.to_excel(output_file, index=False)

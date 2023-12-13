# Add Code
# Import necessary libraries
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob  # Make sure you have TextBlob installed: pip install textblob

# Assuming you have a DataFrame 'dataset' with a column 'reviewText'
# Replace 'your_data.csv' with the actual file path if reading from a file
dataset = pd.read_csv("amazon_reviews.csv")

# Download NLTK resources
nltk.download('stopwords')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Get the set of NLTK stopwords
all_stopwords = set(stopwords.words('english'))


# Function to clean and preprocess text
def preprocess_text(text):
    # Check if the entry is a string
    if isinstance(text, str):
        # Remove non-alphabetic characters
        text = re.sub('[^a-zA-Z]', ' ', text)

        # Convert to lowercase
        text = text.lower()

        # Tokenization and stemming
        text = text.split()
        text = [ps.stem(word) for word in text if word not in all_stopwords]
        text = ' '.join(text)

        return text
    else:
        # If not a string, return an empty string or handle it based on your needs
        return ''


# Apply text preprocessing to each review
dataset['cleaned_review'] = dataset['reviewText'].apply(preprocess_text)

# Apply sentiment analysis using TextBlob
dataset['sentiment'] = dataset['cleaned_review'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Create a histogram to visualize sentiment distribution
plt.figure(figsize=(8, 6))
plt.hist(dataset['sentiment'], bins=30, color='skyblue', edgecolor='black')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Polarity Score')
plt.ylabel('Frequency')
plt.show()

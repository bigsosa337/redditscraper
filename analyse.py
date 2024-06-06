import pandas as pd
import spacy
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the scraped comments
df = pd.read_csv('reddit_comments.csv')

# Preprocess the data
df['cleaned_comment'] = df['Comment'].str.replace(r'http\S+|www.\S+', '', case=False)
df['cleaned_comment'] = df['cleaned_comment'].str.replace('[^a-zA-Z]', ' ')
df['cleaned_comment'] = df['cleaned_comment'].str.lower()

# Sentiment Analysis
df['sentiment'] = df['cleaned_comment'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Visualize Sentiment Distribution
plt.hist(df['sentiment'], bins=20)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Word Cloud
text = ' '.join(df['cleaned_comment'].tolist())
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Topic Modeling
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df['cleaned_comment'])
LDA = LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(dtm)

# Display the top words in each topic
for index, topic in enumerate(LDA.components_):
    print(f'THE TOP 10 WORDS FOR TOPIC #{index}')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print('\n')

# Named Entity Recognition
nlp = spacy.load('en_core_web_sm')
df['entities'] = df['cleaned_comment'].apply(lambda x: [(ent.text, ent.label_) for ent in nlp(x).ents])

# Display the entities in the first few comments
print(df[['Comment', 'entities']].head())

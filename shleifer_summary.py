from transformers import pipeline, AutoTokenizer
import pandas as pd

# Load the scraped comments
df = pd.read_csv('reddit_comments.csv')
comments = df['Comment'].tolist()

# Combine all comments into a single text block
comments_text = " ".join(comments)

# Load the summarization pipeline and tokenizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Function to chunk and summarize text
def chunk_and_summarize(text, summarizer, tokenizer, chunk_size=1024, max_length=150, min_length=50):
    tokens = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    chunk_summaries = []
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarizer(chunk_text, max_length=max_length, min_length=min_length, do_sample=False)
        chunk_summaries.append(summary[0]['summary_text'])
    return " ".join(chunk_summaries)

# Summarize the combined comments text
final_summary = chunk_and_summarize(comments_text, summarizer, tokenizer)

print("Final Summary of Reddit Comments:")
print(final_summary)

# Save the summary to a formatted text file
with open('Reddit_Comments_Summary.txt', 'w', encoding='utf-8') as f:
    f.write("Summary of Reddit Comments\n")
    f.write("===========================\n\n")
    f.write(final_summary)
    f.write("\n\n===========================\n")
    f.write("End of Summary\n")

print("Text file generated successfully.")

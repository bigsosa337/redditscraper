from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

# Load the scraped comments
df = pd.read_csv('reddit_comments.csv')
comments = df['Comment'].tolist()

# Combine all comments into a single text block
comments_text = " ".join(comments)

# Load the T5 tokenizer and model
model_name = "t5-large"  # You can also use "t5-3b" for a larger model if needed
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to chunk and summarize text
def chunk_and_summarize(text, model, tokenizer, chunk_size=512, max_length=150, min_length=50):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    summaries = []
    for chunk in chunks:
        inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)

# Summarize the combined comments text
final_summary = chunk_and_summarize(comments_text, model, tokenizer)

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

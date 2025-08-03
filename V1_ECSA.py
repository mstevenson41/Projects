# Earnings Call Sentiment Analyser
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt

## Prepare Call for Analysis ##
###############################
# Raw transcript
with open("APPL_Earnings_Call_2025_Q2.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Normalize whitespace and remove non-informative lines
import re
text = re.sub(r"\n+", "\n", raw_text)  # Remove multiple newlines

# Match lines that begin with a speaker name (capitalized names), optionally followed by dialogue
pattern = r"(?P<speaker>[A-Z][a-z]+ [A-Z][a-z]+)\n(?P<text>.*?)(?=\n[A-Z][a-z]+ [A-Z][a-z]+\n|$)"  # lookahead for next speaker or end

matches = re.finditer(pattern, text, flags=re.DOTALL)

speaker_data = []

# Filter and collect data
for match in matches:
    speaker = match.group("speaker").strip()
    speech = match.group("text").strip()
    
    if speaker in ["Operator", "Suhasini Chandramouli"]:
        continue

    speaker_data.append({"speaker": speaker, "text": speech})
df = pd.DataFrame(speaker_data)
df.to_csv("apple_2025Q2_transcript_by_speaker.csv", index=False)

###############################
## Analyse Call With FinBert ##
###############################

# Load FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Create sentiment analysis pipeline
finbert_sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

df = pd.read_csv("apple_2025Q2_transcript_by_speaker.csv")

# Limit to first 512 tokens (BERT limit)
def analyze_sentiment(text):
    result = finbert_sentiment(text[:512])[0]
    return pd.Series([result['label'], result['score']])

df[['sentiment', 'confidence']] = df['text'].apply(analyze_sentiment)

# Save results
df.to_csv("apple_2025Q2_finbert_sentiment.csv", index=False)

# Overview of results

print(df['sentiment'].value_counts().reindex(['positive', 'neutral', 'negative']))
print("Positive avg confidence:", df[df['sentiment'] == 'positive']['confidence'].mean())
print("Neutral avg confidence:", df[df['sentiment'] == 'neutral']['confidence'].mean())
print("Negative avg confidence:", df[df['sentiment'] == 'negative']['confidence'].mean())


###############################
## Visualisation of Analysis ##
###############################

# sentiment by speaker
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="speaker", hue="sentiment", order=df['speaker'].value_counts().index)
plt.title("Sentiment by Speaker")
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.xlabel("Speaker")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("sentiment_by_speaker.png")
plt.close()


# Overall sentiment count
sns.countplot(data=df, x="sentiment", order=["positive", "neutral", "negative"])
plt.title("Sentiment Distribution (FinBERT)")
plt.ylabel("Number of Responses")
plt.xlabel("Sentiment")
plt.savefig("sentiment_distribution.png")
plt.close()


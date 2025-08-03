from scraping import Earnings_Call_Scraper
from parser import parse_transcript_to_df
from sentiment import analyze_sentiment
from plotting import SentimentPlotter
from comparative_plotting import plot_comparative_summary
import pandas as pd
import os

def ECSA(url, company, quarter, year):
    Earnings_Call_Scraper(url, company, quarter, year)

    txt_path = f"data/transcripts/{company}_{quarter}_{year}_transcript.txt"
    df = parse_transcript_to_df(txt_path)

    csv_path = f"data/csv/{company}_{quarter}_{year}_by_speaker.csv"
    df.to_csv(csv_path, index=False)

    df[['sentiment', 'confidence']] = df['text'].apply(analyze_sentiment)
    df.to_csv(csv_path.replace(".csv", "_finbert.csv"), index=False)

    plotter = SentimentPlotter(df, company, quarter, year)
    plotter.plot_by_speaker()
    plotter.plot_distribution()
    return df

def Comparative_ECSA(earning_calls):
    comparisons = []
    for call in earning_calls:
        df = ECSA(call["url"], call["company"], call["quarter"], call["year"])

        total = len(df)
        value_counts = df['sentiment'].value_counts()
        results = {
            "company": call["company"],
            "quarter": call["quarter"],
            "year": call["year"],
            "positive_%": round(value_counts.get("positive", 0) / total * 100, 2),
            "neutral_%": round(value_counts.get("neutral", 0) / total * 100, 2),
            "negative_%": round(value_counts.get("negative", 0) / total * 100, 2),
            "overall_counts": total,
            "avg_positive_conf": round(df[df['sentiment'] == 'positive']['confidence'].mean(), 2),
            "avg_neutral_conf": round(df[df['sentiment'] == 'neutral']['confidence'].mean(), 2),
            "avg_negative_conf": round(df[df['sentiment'] == 'negative']['confidence'].mean(), 2)
        }
        comparisons.append(results)

    comparison_df = pd.DataFrame(comparisons)

    # Save CSV
    os.makedirs("comparisons", exist_ok=True)
    csv_path = "comparisons/comparative_sentiment_summary.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")

    # Plot
    plot_comparative_summary(comparison_df)

    return comparison_df
'''
if __name__ == "__main__":
    n = int(input("How many earning calls to analyse? ").strip())
    earning_calls = []
    for i in range(n):
        print(f"\nEnter details for call #{i+1}")
        url = input("Transcript URL: ").strip()
        company = input("Company name: ").strip()
        quarter = input("Quarter (e.g., Q2): ").strip()
        year = input("Year (e.g., 2024): ").strip()
        earning_calls.append({
            "url": url,
            "company": company,
            "quarter": quarter,
            "year": year
        })

    df = Comparative_ECSA(earning_calls)
'''
earning_calls = [
    {
        "url": "https://www.fool.com/earnings/call-transcripts/2024/04/25/microsoft-msft-q3-2024-earnings-call-transcript/",
        "company": "microsoft",
        "quarter": "Q3",
        "year": "2024"
    },
    {
        "url": "https://www.fool.com/earnings/call-transcripts/2024/04/25/alphabet-googl-q1-2024-earnings-call-transcript/",
        "company": "alphabet",
        "quarter": "Q1",
        "year": "2024"
    },
    {
        "url": "https://www.fool.com/earnings/call-transcripts/2024/04/30/amazoncom-amzn-q1-2024-earnings-call-transcript/",
        "company": "amazon",
        "quarter": "Q1",
        "year": "2024"
    },
]

df = Comparative_ECSA(earning_calls)

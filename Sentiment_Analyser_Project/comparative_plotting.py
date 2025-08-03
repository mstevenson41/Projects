import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparative_summary(df):
    df_melt = df.melt(
        id_vars=["company", "quarter", "year"],
        value_vars=["positive_%", "neutral_%", "negative_%"],
        var_name="sentiment",
        value_name="percentage"
    )


    df_melt["label"] = df_melt["company"] + " " + df_melt["quarter"] + " " + df_melt["year"]
    df_melt["sentiment"] = df_melt["sentiment"].str.replace("_%", "").str.capitalize()

    palette = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
    hue_order = ["Positive", "Neutral", "Negative"]

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x="label", y="percentage", hue="sentiment",
                palette=palette, hue_order=hue_order)
    plt.title("Sentiment Comparison Across Earnings Calls")
    plt.ylabel("Sentiment Percentage")
    plt.xlabel("Earnings Call")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs("comparisons", exist_ok=True)
    plt.savefig("comparisons/comparative_sentiment_plot.png")
    plt.close()
    print("Saved plot: comparisons/comparative_sentiment_plot.png")

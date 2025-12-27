from flask import Flask, render_template, request
from nlp_utils import analyze_feedback_list

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    analysis_result = None
    keywords = None
    error = None
    sentiment_stats = None

    if request.method == "POST":
        raw_text = request.form.get("feedback_text", "").strip()

        if not raw_text:
            error = "Please enter at least one feedback comment."
        else:
            feedback_list = raw_text.split("\n")
            df, top_keywords = analyze_feedback_list(feedback_list)

            if df.empty:
                error = "No valid feedback found."
            else:
                analysis_result = df.to_dict(orient="records")
                keywords = top_keywords

                # --- NEW: sentiment statistics for dashboard ---
                total = len(df)
                pos = sum(df["sentiment"].str.lower() == "positive")
                neg = sum(df["sentiment"].str.lower() == "negative")
                neu = total - pos - neg

                sentiment_stats = {
                    "total": total,
                    "positive_count": pos,
                    "negative_count": neg,
                    "neutral_count": neu,
                    "positive_pct": round(pos * 100 / total, 2),
                    "negative_pct": round(neg * 100 / total, 2),
                    "neutral_pct": round(neu * 100 / total, 2),
                }

    return render_template(
        "index.html",
        analysis_result=analysis_result,
        keywords=keywords,
        error=error,
        sentiment_stats=sentiment_stats,
    )

if __name__ == "__main__":
    app.run(debug=True)

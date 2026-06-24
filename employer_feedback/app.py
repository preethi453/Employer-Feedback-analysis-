from flask import Flask, render_template, request
from nlp_utils import analyze_feedback_list

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


@app.route("/analyze", methods=["POST"])
def analyze():

    raw_text = request.form.get(
        "feedback_text",
        ""
    ).strip()

    if not raw_text:

        return render_template(

            "index.html",

            error="Please enter feedback."

        )

    feedback_list = raw_text.split("\n")

    df, top_keywords = analyze_feedback_list(
        feedback_list
    )

    if df.empty:

        return render_template(

            "index.html",

            error="No valid feedback found."

        )

    analysis_result = df.to_dict(
        orient="records"
    )

    total = len(df)

    pos = sum(
        df["sentiment"].str.lower()
        == "positive"
    )

    neg = sum(
        df["sentiment"].str.lower()
        == "negative"
    )

    neu = total - pos - neg

    sentiment_stats = {

        "total": total,

        "positive_count": pos,

        "negative_count": neg,

        "neutral_count": neu,

        "positive_pct": round(
            pos * 100 / total, 2
        ),

        "negative_pct": round(
            neg * 100 / total, 2
        ),

        "neutral_pct": round(
            neu * 100 / total, 2
        ),
    }

    return render_template(

        "result.html",

        analysis_result=analysis_result,

        keywords=top_keywords,

        sentiment_stats=sentiment_stats
    )


if __name__ == "__main__":
    app.run(debug=True)

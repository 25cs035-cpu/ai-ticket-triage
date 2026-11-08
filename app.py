import numpy as np
import pandas as pd
import streamlit as st

from predict import load_models, predict_ticket
from preprocessing import clean_text

st.set_page_config(page_title="AI Ticket Triage", layout="wide")


@st.cache_resource
def get_models():
    """Load the three models once, not on every click."""
    return load_models()


models = get_models()


def predict_batch(texts, threshold):
    """Predict category, department and priority for many tickets at once."""
    cleaned = [clean_text(t) for t in texts]
    out = pd.DataFrame()
    needs_review = np.zeros(len(cleaned), dtype=bool)

    for name, model in models.items():
        probs = model.predict_proba(cleaned)
        out[name] = model.classes_[probs.argmax(axis=1)]
        out[name + "_confidence"] = probs.max(axis=1).round(3)
        needs_review = needs_review | (probs.max(axis=1) < threshold)

    out["needs_review"] = needs_review
    return out


# ---------- Sidebar ----------
st.sidebar.title("Settings")
threshold = st.sidebar.slider(
    "Human review threshold",
    min_value=0.0, max_value=1.0, value=0.40, step=0.05,
    help="If the model's confidence is below this, the ticket is flagged for a human to check.",
)
st.sidebar.caption(
    "Models are modest, especially department, which is the hardest to predict. "
    "Low-confidence tickets should be checked by a person."
)

st.title("AI Customer Support Ticket Triage")

tab1, tab2, tab3 = st.tabs(["Single ticket", "Bulk CSV upload", "Dashboard"])

# ---------- Tab 1: single ticket ----------
with tab1:
    st.subheader("Analyze one ticket")
    text = st.text_area("Paste the customer message here", height=160)

    if st.button("Analyze ticket"):
        if not text.strip():
            st.warning("Please type a customer message first.")
        else:
            result = predict_ticket(text, models, threshold)
            c1, c2, c3 = st.columns(3)
            c1.metric("Category", result["category"],
                      f"{result['category_confidence']:.0%} confident", delta_color="off")
            c2.metric("Department", result["department"],
                      f"{result['department_confidence']:.0%} confident", delta_color="off")
            c3.metric("Priority", result["priority"],
                      f"{result['priority_confidence']:.0%} confident", delta_color="off")

            if result["needs_review"]:
                st.warning("Needs human review: at least one prediction is below the confidence threshold.")
            else:
                st.success("Confident enough to handle automatically.")

# ---------- Tab 2: bulk CSV ----------
with tab2:
    st.subheader("Upload a CSV of tickets")
    st.caption("The CSV needs a 'text' column, or both 'subject' and 'body' columns.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        if "text" in df.columns:
            texts = df["text"].fillna("").astype(str)
        elif "subject" in df.columns and "body" in df.columns:
            texts = df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
        else:
            texts = None
            st.error("Could not find a 'text' column, or 'subject' + 'body' columns.")

        if texts is not None:
            preds = predict_batch(texts.tolist(), threshold)
            results = pd.concat([df.reset_index(drop=True), preds], axis=1)
            st.session_state["results"] = results

            st.success(f"Processed {len(results)} tickets.")
            st.dataframe(results.head(200))
            st.download_button(
                "Download results as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="ticket_predictions.csv",
                mime="text/csv",
            )

# ---------- Tab 3: dashboard ----------
with tab3:
    st.subheader("Dashboard (last uploaded batch)")
    if "results" not in st.session_state:
        st.info("Upload and process a CSV in the 'Bulk CSV upload' tab first.")
    else:
        results = st.session_state["results"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total tickets", len(results))
        m2.metric("High-priority tickets", int((results["priority"] == "high").sum()))
        m3.metric("Needing human review", int(results["needs_review"].sum()))

        d1, d2 = st.columns(2)
        with d1:
            st.write("Tickets by category")
            st.bar_chart(results["category"].value_counts())
        with d2:
            st.write("Tickets by priority")
            st.bar_chart(results["priority"].value_counts())

        st.write("Tickets by department")
        st.bar_chart(results["department"].value_counts())
        

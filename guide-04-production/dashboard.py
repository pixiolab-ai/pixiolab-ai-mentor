import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from app.config import BUSINESS_TZ, DATABASE_URL

# SQLAlchemy wants the driver in the URL: postgresql+psycopg://...
engine = create_engine(DATABASE_URL.replace("postgresql://",
                                            "postgresql+psycopg://", 1))
st.set_page_config(page_title="LLM spend", layout="wide")
st.title("LLM spend")
days = st.slider("Days to show", 7, 90, 30)


@st.cache_data(ttl=60)
def load(days: int) -> pd.DataFrame:
    df = pd.read_sql(text(
        "SELECT created_at, user_id, feature, model, fallback, cost_usd"
        " FROM llm_usage"
        " WHERE created_at > now() - (:days * interval '1 day')"),
        engine, params={"days": days})
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["created_at"] = df["created_at"].dt.tz_convert(BUSINESS_TZ)
    df["cost_usd"] = pd.to_numeric(df["cost_usd"])  # NULL becomes NaN
    return df


df = load(days)
if df.empty:
    st.info("No usage rows yet. Make a few tracked calls first.")
    st.stop()

today = pd.Timestamp.now(tz=BUSINESS_TZ).normalize()  # business midnight
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Spend, last {days} days", f"${df.cost_usd.sum():,.2f}")
c2.metric("Spend today", f"${df[df.created_at >= today].cost_usd.sum():,.2f}")
c3.metric("Calls", f"{len(df):,}")
c4.metric("Calls with no price", f"{df.cost_usd.isna().sum():,}")

st.subheader("By day")
st.bar_chart(df.set_index("created_at").cost_usd.resample("D").sum())
left, right = st.columns(2)
left.subheader("By feature")
left.bar_chart(df.groupby("feature").cost_usd.sum())
right.subheader("By model (fallback included)")
right.bar_chart(df.groupby("model").cost_usd.sum())
st.subheader("Top 20 users")
st.dataframe(df.groupby("user_id").cost_usd.sum().nlargest(20))

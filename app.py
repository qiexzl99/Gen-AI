import os
import json
import re
from typing import List, Dict


import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# 0) Environment & Page Config
# -----------------------------
load_dotenv()
st.set_page_config(page_title="EV BI + GenAI Assistant", page_icon="🔋", layout="wide")

# Prefer st.secrets first; fall back to env var
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# -----------------------------
# 1) Imports for LangChain (new style)
# -----------------------------
# pip install -U langchain langchain-openai langchain-community
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# -----------------------------
# 2) Helpers & Data Loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def summarize_dataframe(df: pd.DataFrame, max_rows: int = 200,use_full: bool = False, eff_metric: str = "km per SOC%") -> Dict:
    """Summarize the dataframe into a compact JSON for LLM context.
    - Exact > whole-word > regex (with excludes) column detection
    - Prefer station_location as region grouping (fallback to city-like)
    - Derive charging_efficiency using your derived cols when present
    """
    info: Dict = {"columns": list(df.columns), "rows": int(len(df))}

    # Numeric synopsis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = (
        df[numeric_cols].describe().T.reset_index().rename(columns={"index": "column"})
        if numeric_cols else pd.DataFrame()
    )

    summary: Dict = {}

    # ------------ robust column detection helpers ------------
    def _norm_tokens(s: str):
        return re.findall(r"[a-z0-9]+", str(s).lower())

    def get_col(
        df: pd.DataFrame,
        exact: List[str] = None,
        words_all: List[str] = None,
        regex_any: List[str] = None,
        exclude_regex: List[str] = None,
    ):
        exact = exact or []
        words_all = words_all or []
        regex_any = regex_any or []
        exclude_regex = exclude_regex or []

        def _excluded(name: str) -> bool:
            lc = str(name).lower()
            return any(re.search(p, lc) for p in exclude_regex)

        # 1) exact (case-insensitive)
        name_map = {c.lower(): c for c in df.columns}
        for x in exact:
            if x.lower() in name_map and not _excluded(name_map[x.lower()]):
                return name_map[x.lower()]

        # 2) whole-word token match  (避免把 capacity 误判成 city)
        for c in df.columns:
            if _excluded(c):
                continue
            toks = set(_norm_tokens(c))
            if all(w.lower() in toks for w in words_all):
                return c

        # 3) regex any
        for c in df.columns:
            if _excluded(c):
                continue
            lc = str(c).lower()
            if any(re.search(p, lc) for p in regex_any):
                return c
        return None

    # ------------ map your full schema ------------
    col_user_id   = get_col(df, exact=["user_id"], words_all=["user","id"])
    col_vehicle   = get_col(df, exact=["vehicle_model"], words_all=["vehicle","model"])
    col_batt_cap  = get_col(df, exact=["battery_capacity_kwh"], regex_any=[r"battery.*kwh"])
    col_station   = get_col(df, exact=["station_id"], words_all=["station","id"])

    # Region: prefer station_location; avoid 'capacity' false positive
    col_station_location = get_col(df, exact=["station_location"], regex_any=[r"station[_\s]*location"])
    col_city_like        = get_col(df, exact=["city"], words_all=["city"], exclude_regex=[r"capacity"])  # optional
    col_region           = col_station_location or col_city_like  # 用于分组

    # Timestamps & time features
    col_start_time = get_col(df, exact=["start_time"], regex_any=[r"start[_\s]*time"])
    col_end_time   = get_col(df, exact=["end_time"],   regex_any=[r"end[_\s]*time"])
    col_duration   = get_col(df, exact=["duration_h"], words_all=["duration"])
    col_start_hour = get_col(df, exact=["start_hour"], words_all=["hour"])
    col_start_date = get_col(df, exact=["start_date"], words_all=["date"])
    col_start_wd   = get_col(df, exact=["start_weekday"], regex_any=[r"weekday", r"day of week"])

    # Energy / power / cost
    col_energy_kwh   = get_col(df, exact=["energy_kwh"])
    col_rate_kw      = get_col(df, exact=["rate_kw"])
    col_cost_usd     = get_col(df, exact=["cost_usd"])
    col_cost         = get_col(df, exact=["cost"]) or col_cost_usd
    col_cost_per_kwh = get_col(df, exact=["cost_per_kwh"], regex_any=[r"cost per kwh", r"price/?kwh"])

    # Categorical contexts
    col_time_of_day = get_col(df, exact=["time_of_day"], regex_any=[r"time of day"])
    col_day_of_week = get_col(df, exact=["day_of_week"], regex_any=[r"day of week"])
    col_charger_type= get_col(df, exact=["Charger Type"], words_all=["charger","type"])
    col_user_type   = get_col(df, exact=["User Type"],    words_all=["user","type"])

    # SOC & derived
    col_soc_start   = get_col(df, exact=["soc_start_pct"], regex_any=[r"soc.*start"])
    col_soc_end     = get_col(df, exact=["soc_end_pct"],   regex_any=[r"soc.*end"])
    col_soc_delta   = get_col(df, exact=["soc_delta_pct"], regex_any=[r"soc.*delta"])
    col_soc_rate    = get_col(df, exact=["soc_increase_rate_pct_per_h"], regex_any=[r"increase[_\s]*rate", r"per[_\s]*h"])
    col_kwh_per_soc = get_col(df, exact=["kwh_per_soc_pct"])

    # Distance / temperature / age
    col_distance_km = get_col(
        df,
        exact=["Distance Driven (since last charge) (km)"],
        regex_any=[r"distance.*since.*charge", r"distance.*km", r"mileage"]
    )
    col_temp        = get_col(df, exact=["Temperature (掳C)", "Temperature (°C)", "temperature"], regex_any=[r"temp"])
    col_vehicle_age = get_col(df, exact=["Vehicle Age (years)", "vehicle_age_years"], regex_any=[r"vehicle.*age"])

    # Downsample for context
    df_small = df.copy()
    if (not use_full) and (len(df_small) > max_rows):
        df_small = df_small.sample(max_rows, random_state=42)

    # ------------ derived metrics ------------
    # soc_increase_pct
    if col_soc_delta and col_soc_delta in df_small.columns:
        df_small["soc_increase_pct"] = pd.to_numeric(df_small[col_soc_delta], errors="coerce")
    elif col_soc_start and col_soc_end:
        with np.errstate(divide='ignore', invalid='ignore'):
            df_small["soc_increase_pct"] = (
                pd.to_numeric(df_small[col_soc_end], errors="coerce")
                - pd.to_numeric(df_small[col_soc_start], errors="coerce")
            )
    else:
        df_small["soc_increase_pct"] = np.nan

    # charging_efficiency: distance per SOC pct (fallback: 1 / kwh_per_soc_pct)
    if col_distance_km and "soc_increase_pct" in df_small.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df_small["charging_efficiency"] = (
                pd.to_numeric(df_small[col_distance_km], errors="coerce")
                / df_small["soc_increase_pct"].replace(0, np.nan)
            )
    elif col_kwh_per_soc:
        with np.errstate(divide='ignore', invalid='ignore'):
            df_small["charging_efficiency"] = 1 / pd.to_numeric(df_small[col_kwh_per_soc], errors="coerce").replace(0, np.nan)
    else:
        df_small["charging_efficiency"] = np.nan

    # --- unified efficiency value based on user's choice ---
    eff_value = pd.Series(np.nan, index=df_small.index, dtype="float64")

    if eff_metric == "km per SOC%":
        if col_distance_km is not None and "soc_increase_pct" in df_small:
            with np.errstate(divide='ignore', invalid='ignore'):
                eff_value = pd.to_numeric(df_small[col_distance_km], errors="coerce") / df_small[
                    "soc_increase_pct"].replace(0, np.nan)

    elif eff_metric == "kWh per USD":
        if col_energy_kwh and (col_cost or col_cost_usd):
            e = pd.to_numeric(df_small[col_energy_kwh], errors="coerce")
            cost_col = col_cost if col_cost else col_cost_usd
            c = pd.to_numeric(df_small[cost_col], errors="coerce")
            with np.errstate(divide='ignore', invalid='ignore'):
                eff_value = e / c.replace(0, np.nan)

    elif eff_metric == "km per kWh":
        if col_distance_km and col_energy_kwh:
            d = pd.to_numeric(df_small[col_distance_km], errors="coerce")
            e = pd.to_numeric(df_small[col_energy_kwh], errors="coerce")
            with np.errstate(divide='ignore', invalid='ignore'):
                eff_value = d / e.replace(0, np.nan)

    df_small["eff_value"] = eff_value

    # ------------ 1) Efficiency leaderboard by region ------------
    region_label = col_region if col_region else col_station_location  # 仍优先 station_location
    if col_vehicle and region_label and ("eff_value" in df_small):
        eff = (
            df_small[[col_vehicle, region_label, "eff_value"]]
            .dropna(subset=["eff_value"])
            .groupby([region_label, col_vehicle])
            .eff_value.mean()
            .reset_index()
        )
        top_eff = (
            eff.sort_values([region_label, "eff_value"], ascending=[True, False])
            .groupby(region_label)
            .head(5)
        )
        summary["top_efficiency_by_group"] = top_eff.round(4).to_dict(orient="records")
        summary["efficiency_group_field"] = region_label
        summary["efficiency_metric"] = eff_metric

    # ------------ 2) Station usage frequency ------------
    if col_station:
        usage = (
            df_small[col_station]
            .value_counts()
            .head(20)
            .reset_index()
            .rename(columns={"index": str(col_station), col_station: "count"})
        )
        summary["station_usage_top"] = usage.to_dict(orient="records")

    # ------------ 3) Behavior patterns (time-of-day / hour / weekday) ------------
    if col_time_of_day:
        tod = df_small[col_time_of_day].value_counts().reset_index().rename(
            columns={"index": str(col_time_of_day), col_time_of_day: "count"}
        )
        summary["time_of_day_distribution"] = tod.to_dict(orient="records")
    elif col_start_hour and col_start_hour in df_small.columns:
        hour_series = pd.to_numeric(df_small[col_start_hour], errors="coerce")
        hour_dist = hour_series.value_counts().sort_index().reset_index()
        hour_dist.columns = ["hour", "count"]
        summary["hourly_distribution"] = hour_dist.to_dict(orient="records")
    elif col_start_time and col_start_time in df_small.columns:
        try:
            ts = pd.to_datetime(df_small[col_start_time], errors="coerce")
            hour_dist = ts.dt.hour.value_counts().sort_index().reset_index()
            hour_dist.columns = ["hour", "count"]
            summary["hourly_distribution"] = hour_dist.to_dict(orient="records")
        except Exception:
            pass

    # weekday: prefer start_weekday, fallback to day_of_week
    if col_start_wd and col_start_wd in df_small.columns:
        wd_series = df_small[col_start_wd]
    elif col_day_of_week and col_day_of_week in df_small.columns:
        wd_series = df_small[col_day_of_week]
    else:
        wd_series = None

    if wd_series is not None:
        order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        wd = wd_series.value_counts().reindex(order, fill_value=0).reset_index()
        if not wd.empty:
            wd.columns = ["weekday", "count"]
            summary["weekday_distribution"] = wd.to_dict(orient="records")

    # Numeric description (rounded)
    if not desc.empty:
        desc_rounded = desc.copy()
        for c in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            if c in desc_rounded.columns:
                desc_rounded[c] = desc_rounded[c].round(3)
        summary["numeric_description"] = desc_rounded.to_dict(orient="records")

    # for debugging/LLM context
    summary["_detected_cols"] = {
        "user_id": col_user_id,
        "vehicle_model": col_vehicle,
        "battery_capacity_kwh": col_batt_cap,
        "station_id": col_station,
        "station_location": col_station_location,
        "region_used": col_region,
        "start_time": col_start_time,
        "end_time": col_end_time,
        "duration_h": col_duration,
        "start_hour": col_start_hour,
        "start_date": col_start_date,
        "start_weekday": col_start_wd,
        "energy_kwh": col_energy_kwh,
        "rate_kw": col_rate_kw,
        "cost": col_cost,
        "cost_usd": col_cost_usd,
        "cost_per_kwh": col_cost_per_kwh,
        "time_of_day": col_time_of_day,
        "day_of_week": col_day_of_week,
        "soc_start_pct": col_soc_start,
        "soc_end_pct": col_soc_end,
        "soc_delta_pct": col_soc_delta,
        "soc_increase_rate_pct_per_h": col_soc_rate,
        "kwh_per_soc_pct": col_kwh_per_soc,
        "distance_km": col_distance_km,
        "temperature": col_temp,
        "vehicle_age_years": col_vehicle_age,
        "charger_type": col_charger_type,
        "user_type": col_user_type,
    }

    info["summary"] = summary
    return info

# -----------------------------
# 3) UI - Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Model", options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)

eff_metric = st.sidebar.selectbox(
    "Efficiency metric",
    ["km per SOC%", "kWh per USD", "km per kWh"],
    index=0,
    help="Choose the definition to match your Excel."
)

max_context_rows = st.sidebar.slider("Rows to summarize for LLM context", 50, 1000, 200, step=50)
use_full_for_llm = st.sidebar.checkbox(
    "Use FULL dataset for LLM aggregates (no downsampling)",
    value=True
)

def safe_read(path: str) -> pd.DataFrame:
    try:
        return load_csv(path)
    except FileNotFoundError:
        st.warning(f"CSV not found at '{path}'. Upload a file below.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame()

# Data source choice
use_uploaded = st.sidebar.checkbox("Use uploaded CSV (instead of local file)", value=False)

st.title("🔋 EV Charging Data Assistant – BI + GenAI")
st.caption("An AI-augmented analytics app that blends Tableau visualizations with LLM insights.")

uploaded_df = pd.DataFrame()
if use_uploaded:
    up = st.file_uploader("Upload your EV CSV", type=["csv"])
    if up is not None:
        try:
            uploaded_df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Upload parse error: {e}")
else:
    uploaded_df = safe_read("ev_charging_patterns_cleaned.csv")

if uploaded_df.empty:
    st.info("No data loaded yet. Please upload a CSV or place 'ev_charging_patterns.csv' next to this app.")
else:
    st.subheader("📊 Data Preview")
    st.dataframe(uploaded_df.head(10), use_container_width=True)

# -----------------------------
# 4) Tableau Embed (in-app iframe)
# -----------------------------
st.markdown("---")
st.subheader("🌐 Interactive Tableau Dashboard (Embedded)")

def normalize_tableau_url(url: str) -> str:
    # Ensure showVizHome=no for cleaner embed
    if "?" in url:
        if "showVizHome" not in url:
            return url + "&:showVizHome=no"
        return url
    return url + "?:showVizHome=no"

with st.expander("Embed a Tableau Public View", expanded=True):
    tableau_url = st.text_input(
        "Tableau Public View URL",
        value="https://public.tableau.com/views/YourWorkbook/YourSheet",
        help="Paste the share link from Tableau Public (Sheets or Dashboards)."
    )
    embed_w = st.slider("Embed width (px)", min_value=600, max_value=2400, value=1200, step=20)
    embed_h = st.slider("Embed height (px)", min_value=500, max_value=1400, value=820, step=20)

    if tableau_url.strip():
        st.components.v1.iframe(
            src=normalize_tableau_url(tableau_url.strip()),
            width=embed_w,
            height=embed_h,
            scrolling=False
        )

# -----------------------------
# 5) LangChain Chat with Memory (multi-turn)
# -----------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages: List = []  # list of dicts {role, content}

# System instruction includes how to read the compact summary
SYSTEM_PROMPT = (
    "You are an expert EV charging data analyst. You answer in clear, structured English (or the user's language). "
    "You receive a COMPACT_DATA_SUMMARY extracted from a CSV. "
    "All aggregates may be computed on the FULL dataset when meta.use_full is true. "
    "Only summarize and explain the provided aggregates and tables (e.g., top_efficiency_by_group). "
    "If a numeric value is not explicitly present in the summary, do NOT invent it; say 'not available from summary'. "
    "When uncertain, state assumptions. "
    "Prioritize: (1) charging efficiency by vehicle model across regions, (2) station usage and regional patterns, "
    "(3) user behavior (time-of-day/hour). When useful, propose follow-up analyses, KPIs, and Tableau chart suggestions. "
    "Keep answers concise but insightful."
)

# Build LLM
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it to .env or st.secrets.")
else:
    llm = ChatOpenAI(model=model_name, temperature=0.2, openai_api_key=OPENAI_API_KEY)

# Compose context summary for the model
data_context: Dict = {}
if not uploaded_df.empty:
    data_context = summarize_dataframe(
        uploaded_df,
        max_rows=max_context_rows,         # 仅在 use_full=False 时生效
        use_full=use_full_for_llm,         # ★ 全量统计
        eff_metric=eff_metric              # ★ 与 Excel 口径对齐
    )

st.markdown("---")
st.subheader("🔎 Verification table (from FULL data)")

if not uploaded_df.empty:
    dfv = uploaded_df.copy()

    # 复用与 summarize_dataframe 相同的效率口径
    def _compute_eff_value(df_base: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=df_base.index, dtype="float64")
        if eff_metric == "km per SOC%":
            if "soc_end_pct" in df_base.columns and "soc_start_pct" in df_base.columns and "Distance Driven (since last charge) (km)" in df_base.columns:
                inc = pd.to_numeric(df_base["soc_end_pct"], errors="coerce") - pd.to_numeric(df_base["soc_start_pct"], errors="coerce")
                with np.errstate(divide='ignore', invalid='ignore'):
                    out = pd.to_numeric(df_base["Distance Driven (since last charge) (km)"], errors="coerce") / inc.replace(0, np.nan)
        elif eff_metric == "kWh per USD":
            if "energy_kwh" in df_base.columns and ("cost" in df_base.columns or "cost_usd" in df_base.columns):
                e = pd.to_numeric(df_base["energy_kwh"], errors="coerce")
                cost_col = "cost" if "cost" in df_base.columns else "cost_usd"
                c = pd.to_numeric(df_base[cost_col], errors="coerce")
                with np.errstate(divide='ignore', invalid='ignore'):
                    out = e / c.replace(0, np.nan)
        elif eff_metric == "km per kWh":
            if "energy_kwh" in df_base.columns and "Distance Driven (since last charge) (km)" in df_base.columns:
                d = pd.to_numeric(df_base["Distance Driven (since last charge) (km)"], errors="coerce")
                e = pd.to_numeric(df_base["energy_kwh"], errors="coerce")
                with np.errstate(divide='ignore', invalid='ignore'):
                    out = d / e.replace(0, np.nan)
        return out

    dfv["eff_value"] = _compute_eff_value(dfv)

    group_field = data_context.get("summary", {}).get("efficiency_group_field", None)
    if group_field and ("vehicle_model" in dfv.columns):
        verif = (
            dfv[[group_field, "vehicle_model", "eff_value"]]
            .dropna(subset=["eff_value"])
            .groupby([group_field, "vehicle_model"])
            .eff_value.mean()
            .reset_index()
        )
        verif_top = (
            verif.sort_values([group_field, "eff_value"], ascending=[True, False])
            .groupby(group_field, as_index=False).head(5)
        )
        st.dataframe(verif_top.round(4), use_container_width=True)
    else:
        st.info("Missing group field or vehicle_model; please check detected columns.")

# Chat UI
st.markdown("---")
st.subheader("💬 Chat with the EV Data Assistant")
st.caption("Ask questions like: *Which stations are most utilized in London?* or *Which models have the highest efficiency by city?*")

# Show history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Starter quick actions
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Ask: Top efficient models by city"):
        st.session_state._pending_user = "Which vehicle models show the highest charging efficiency by city? Please list top 3 per city."
with c2:
    if st.button("Ask: Station usage patterns"):
        st.session_state._pending_user = "Which charging stations are used most frequently, and what regional patterns exist?"
with c3:
    if st.button("Ask: Behavior patterns"):
        st.session_state._pending_user = "What user behavior patterns are visible (time of day or hourly)? Any recommendations to shift demand?"

user_query = st.chat_input("Type your question about the dataset…")
if "_pending_user" in st.session_state and st.session_state._pending_user:
    user_query = st.session_state._pending_user
    st.session_state._pending_user = ""

if user_query:
    # Append user message
    st.session_state.chat_messages.append({"role": "user", "content": user_query})

    # Assemble messages for LLM
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add compact data context as an initial assistant tool-message style
    if data_context:
        data_blob = json.dumps(data_context, ensure_ascii=False)
        messages.append(AIMessage(content=f"COMPACT_DATA_SUMMARY:\n{data_blob}"))

    # Add history
    for m in st.session_state.chat_messages:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))

    # Call LLM
    if OPENAI_API_KEY:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    ai_resp = llm.invoke(messages)
                    text = ai_resp.content
                except Exception as e:
                    text = f"⚠️ LLM error: {e}"
                st.markdown(text)
        # Save assistant reply
        st.session_state.chat_messages.append({"role": "assistant", "content": text})

# -----------------------------
# 6) Optional: On-demand AI report (single-shot)
# -----------------------------
st.markdown("---")
st.subheader("🧠 One-Click AI Report")
st.caption("Generates a compact narrative using the summarized data context.")
if st.button("Generate AI Report"):
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY.")
    elif uploaded_df.empty:
        st.warning("Please load a dataset first.")
    else:
        base_q = (
            "Create a short, structured report with bullet points covering: "
            "(1) top efficient vehicle models by city (with caveats), "
            "(2) station usage patterns and outliers, (3) user behavior timing patterns, "
            "(4) suggested Tableau charts and next analyses."
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.append(AIMessage(content=f"COMPACT_DATA_SUMMARY:\n{json.dumps(data_context, ensure_ascii=False)}"))
        messages.append(HumanMessage(content=base_q))
        with st.spinner("Generating report…"):
            try:
                res = llm.invoke(messages)
                st.success("Done.")
                st.markdown(res.content)
            except Exception as e:
                st.error(f"LLM error: {e}")

# -----------------------------
# 7) Footer
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit · LangChain (new-style) · Tableau Public")

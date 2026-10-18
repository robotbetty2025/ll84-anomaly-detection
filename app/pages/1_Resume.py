import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
import streamlit as st
import content as C

st.set_page_config(page_title="Resume", page_icon="📄", layout="wide")
st.title("Resume")
if C.PLACEHOLDERS_REMAIN:
    st.warning("Placeholder content: edit app/content.py with your real positions, dates and skill levels.")

pdf = Path(__file__).resolve().parents[1] / C.RESUME_PDF
if pdf.exists():
    st.download_button("Download resume (PDF)", pdf.read_bytes(), file_name=f"{C.NAME.replace(' ', '_')}_Resume.pdf", mime="application/pdf")
else:
    st.info("Add your resume at app/assets/resume.pdf to enable the download button.")

st.header("Education")
st.dataframe(pd.DataFrame(C.EDUCATION).rename(columns={"degree": "Degree", "school": "Institution", "status": "Status"}), hide_index=True)

st.header("Career timeline")
tl = pd.DataFrame(C.EXPERIENCE)
tl["Start"], tl["Finish"] = pd.to_datetime(tl.start), pd.to_datetime(tl.end)
tl["Role"] = tl.title + " | " + tl.org
fig = px.timeline(tl, x_start="Start", x_end="Finish", y="Role", color="area", hover_data={"Start": False, "Finish": False})
fig.update_yaxes(autorange="reversed", title=""); fig.update_layout(height=140 + 70 * len(tl), legend_title="Focus area")
st.plotly_chart(fig)

st.header("Experience")
for e in C.EXPERIENCE:
    st.markdown(f"**{e['title']}**, {e['org']} ({e['start'][:4]} to {e['end'][:4]})")
    for b in e["bullets"]:
        st.markdown(f"- {b}")

st.header("Skills")
sk = pd.DataFrame({"Skill": list(C.SKILLS), "Level (1-5)": list(C.SKILLS.values())}).sort_values("Level (1-5)")
fig2 = px.bar(sk, x="Level (1-5)", y="Skill", orientation="h", range_x=[0, 5], color="Level (1-5)", color_continuous_scale="Teal")
fig2.update_layout(height=60 + 42 * len(sk), coloraxis_showscale=False, yaxis_title="")
st.plotly_chart(fig2)

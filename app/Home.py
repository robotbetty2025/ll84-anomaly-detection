import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
import content as C

st.set_page_config(page_title=f"{C.NAME} | Portfolio", page_icon="🏢", layout="wide")
if C.PLACEHOLDERS_REMAIN:
    st.warning("Placeholder content: edit app/content.py (and add assets/headshot.png) before sharing this page.")

left, right = st.columns([1, 3])
with left:
    p = Path(__file__).parent / C.HEADSHOT
    if p.exists():
        st.image(str(p))
    else:
        st.info("Add a headshot at app/assets/headshot.png")
with right:
    st.title(C.NAME)
    st.subheader(C.HEADLINE)
    st.write(C.SUMMARY)
    st.write("**Credentials:** " + ", ".join(C.CREDENTIALS))
    cols = st.columns(len(C.LINKS) + 1)
    for col, (label, url) in zip(cols, C.LINKS.items()):
        col.link_button(label, url)
    if C.EMAIL:
        cols[-1].write(C.EMAIL)

st.divider()
st.markdown("### Explore this portfolio")
a, b, c = st.columns(3)
a.markdown("**Resume**\n\nEducation, experience, a career timeline and a downloadable PDF.")
b.markdown("**Projects**\n\nMachine learning, analytics and sustainability work.")
c.markdown("**Featured: Building Energy Anomaly Detector**\n\nA live model that ranks NYC buildings by unusual energy use, "
           "with explanations. Use the sidebar to open it.")

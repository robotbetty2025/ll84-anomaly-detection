import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import content as C

st.set_page_config(page_title="Projects", page_icon="🗂️", layout="wide")
st.title("Projects")
if C.PLACEHOLDERS_REMAIN:
    st.warning("Placeholder content: edit app/content.py to add your real projects and repository links.")
cols = st.columns(2)
for i, p in enumerate(C.PROJECTS):
    with cols[i % 2].container(border=True):
        st.markdown(f"#### {p['title']}")
        st.caption(p["kind"])
        st.write(p["blurb"])
        if p["tags"]:
            st.write(" ".join(f"`{t}`" for t in p["tags"]))
        if p["page"]:
            try:
                st.page_link(p["page"], label="Open the live project")
            except Exception:       # page not registered (e.g. running this page on its own)
                st.caption("Open it from the sidebar menu.")
        if p["repo"]:
            st.link_button("Repository", p["repo"])

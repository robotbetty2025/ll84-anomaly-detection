"""Editable site content. Set PLACEHOLDERS_REMAIN = False once everything below is final."""
PLACEHOLDERS_REMAIN = False         # shows an "edit me" banner on pages while True

NAME = "Julie Bounds"
HEADLINE = "Technical program manager and sustainability professional moving into data science: building, real estate and energy analytics"
SUMMARY = (
    "I have 15+ years of experience leading sustainability, construction and technology programs, from large public "
    "school building programs to enterprise smart-building deployments. I hold LEED AP BD+C and PMP credentials and "
    "am pursuing an MBA and an MSc in Data Science at Eastern University. My focus is energy analytics for buildings "
    "and real estate: finding where buildings use more (or less) energy than they should, and explaining why."
)
CREDENTIALS = ["LEED AP BD+C", "PMP"]
LINKS = {
    "GitHub": "https://github.com/robotbetty2025",
}
EMAIL = "jmeadors10@gmail.com"

EDUCATION = [
    {"degree": "MBA and MSc in Data Science", "school": "Eastern University", "status": "In progress"},
    {"degree": "B.S. Engineering", "school": "University of Mississippi", "status": "Completed"},
]
# Dates drive the timeline chart on the Resume page.
EXPERIENCE = [
    {"title": "Senior Project Manager", "org": "Atrius, Acuity Brands", "start": "2023-11", "end": "2026-09",
     "area": "Smart buildings and enterprise technology deployment",
     "bullets": [
         "Led enterprise-scale technology deployment programs valued at over $10M across commercial and data center environments.",
         "Built project KPIs, forecasts, schedules and risk plans to improve delivery outcomes across multiple concurrent projects.",
         "Drove process improvements to raise documentation quality and efficiency across distributed teams.",
     ]},
    {"title": "LEED Project Manager", "org": "EcoPreserve", "start": "2022-11", "end": "2023-11",
     "area": "Sustainability / LEED",
     "bullets": [
         "Managed sustainability and compliance work across multiple commercial projects throughout the United States.",
         "Analyzed project data on water use, materials, indoor air quality, transportation and environmental performance.",
         "Built documentation workflows and reporting systems that improved submission accuracy.",
     ]},
    {"title": "Program Manager", "org": "ACR Engineering", "start": "2010-01", "end": "2022-11",
     "area": "MEP design, commissioning and sustainability programs",
     "bullets": [
         "Led programs supporting over 80 commercial and institutional projects in Austin ISD's $1.5B bond program.",
         "Built operational processes, tracking systems and reporting structures that improved consistency across programs.",
         "Trained and mentored new employees on systems, processes and technical standards.",
     ]},
]
SKILLS = {                          # 1 (learning) to 5 (expert)
    "Technical program management": 5, "LEED / sustainability programs": 5, "Project management (PMP)": 5,
    "Excel / Smartsheet / KPI reporting": 4, "SQL": 3, "Tableau / Power BI": 3,
    "Python (pandas, NumPy, scikit-learn)": 3, "Machine learning (LightGBM, Isolation Forest)": 3,
    "Model explainability (SHAP)": 3, "Streamlit / Plotly": 3,
}
RESUME_PDF = "assets/resume.pdf"    # add your PDF here to enable the download button
HEADSHOT = "assets/headshot.png"    # add a photo here

PROJECTS = [
    {"title": "Building Energy Anomaly Detection (capstone)", "kind": "Machine learning",
     "blurb": "Ranks NYC commercial and multifamily buildings by how far their energy use departs from comparable buildings, "
              "with explanations for each flag. Live model and interactive explorer.",
     "tags": ["LightGBM", "Isolation Forest", "SHAP", "Streamlit"], "page": "pages/3_Anomaly_Detector.py", "repo": ""},
    {"title": "Austin ISD Bond Program", "kind": "Professional",
     "blurb": "Led operational and sustainability programs supporting over 80 commercial and institutional projects "
              "in a $1.5B school bond program.",
     "tags": ["Program management", "Sustainability", "Green building"], "page": None, "repo": ""},
    {"title": "Enterprise Technology Deployments", "kind": "Professional",
     "blurb": "Led deployment programs valued at over $10M across commercial and data center environments, "
              "with KPI tracking, forecasting and risk management.",
     "tags": ["Smart buildings", "KPIs", "Risk management"], "page": None, "repo": ""},
]

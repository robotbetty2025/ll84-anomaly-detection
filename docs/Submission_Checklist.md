# Submission Checklist (Week 7)

Do these in order. Items marked ASK CLAUDE can be done by me once you send the input.

1. [ ] Open `reports/expert_audit_50_buildings.xlsx`, classify all 50 rows with the dropdown (do not open the Key sheet until done).
2. [ ] ASK CLAUDE: run `scripts/05_audit_results.py` and `scripts/06_build_report.py` to put the audit into the PDF.
3. [ ] Fill in `app/content.py` (bio, resume, projects, links); add `app/assets/headshot.png` and `resume.pdf`; set `PLACEHOLDERS_REMAIN = False`.
4. [ ] Create a GitHub repo and push the project folder (the `.gitignore` already keeps the app data files).
5. [ ] Deploy at share.streamlit.io: repo, branch main, main file `app/Home.py`.
6. [ ] Put the GitHub and Streamlit URLs in `docs/report_config.json`.
7. [ ] Record the video using `docs/Video_Script_and_Shot_List.md`; upload it and copy the link into `report_config.json`.
8. [ ] ASK CLAUDE: re-run script 06 so the PDF has all three links.
9. [ ] Read the PDF once for anything that does not sound like you.
10. [ ] Upload to Google Drive and Brightspace: Final_Project_Overview.pdf, video, GitHub link, Streamlit link, AI_USAGE_LOG.md.

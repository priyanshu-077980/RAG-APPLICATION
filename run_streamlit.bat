@echo off
title ChatGPT RAG Assistant (Streamlit)
echo ====================================================
echo Starting ChatGPT RAG Assistant on http://localhost:8501
echo ====================================================
call .\.venv\Scripts\python.exe -m streamlit run app.py
pause


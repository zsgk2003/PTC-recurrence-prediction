@echo off
chcp 65001 >nul
echo ============================================================
echo Thyroid Cancer Recurrence Predictor - Streamlit App
echo ============================================================
echo.
echo [1/2] Checking dependencies...
python -m pip install -q -r requirements.txt
echo.
echo [2/2] Launching Streamlit app...
echo Browser will open automatically at http://localhost:8501
echo Press Ctrl+C to stop the server
echo ============================================================
python -m streamlit run app.py
pause

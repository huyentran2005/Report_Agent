"""Run with: streamlit run app.py"""
import runpy
import sys
from pathlib import Path

project = Path(__file__).parent
sys.path.insert(0, str(project / "src"))
runpy.run_path(str(project / "src" / "streamlit_app.py"), run_name="__main__")

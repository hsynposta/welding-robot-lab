"""ENG3032 - Robotics Laboratory v2 (9 modules).

The original single-file HTML/JS lab is embedded verbatim — kinematics, inverse
kinematics, scenarios, forward/inverse dynamics, PID, trajectory, drive
selection and energy modules all run exactly as authored, including the
built-in dark-mode toggle, tabs, canvas animations and the self-check quiz.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ENG3032 · Robotics Lab v2", page_icon="🧪",
                   layout="wide", initial_sidebar_state="collapsed")

# let the embedded lab use the full viewport width
st.markdown("""<style>
  .block-container {padding: 0.6rem 0.8rem 0; max-width: 1560px; margin: 0 auto;}
</style>""", unsafe_allow_html=True)

HTML = (Path(__file__).parent.parent / "assets" / "robotics_lab_v2.html"
        ).read_text(encoding="utf-8")

components.html(HTML, height=1750, scrolling=True)

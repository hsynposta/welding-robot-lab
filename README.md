# ENG3032 · Welding Robot Lab

Interactive Streamlit app for a single-joint welding robot arm:
kinematics → dynamics → power → energy, identical to the course MATLAB model
(trapezoidal integration, 501 samples). Includes a second page with a
multi-link FABRIK inverse-kinematics simulation lab.

## Run locally

```bash
pip install -r requirements.txt
streamlit run welding_robot_lab.py
```

## Pages

- **Welding Robot Lab** — two independently parameterised scenarios
  (m, l, C and motion profile per scenario), synchronized schematic + chart
  animation, live equations, optional annual fleet economics.
- **Simulation Lab** — 1/2/3-link planar arms, joint control,
  click-target FABRIK IK, and motion playback.

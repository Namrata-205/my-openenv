import streamlit as st
import requests

BASE_URL = "https://dcode7-openenv.hf.space"

st.title("ATC RL Environment")

# Let user choose task
TASK = st.text_input("Enter Task Name:", "wake_turbulence")

# Initialize session state
if "state" not in st.session_state:
    st.session_state["state"] = None
    st.session_state["step"] = 0

# Helper function to get aircraft/entities list dynamically
def get_aircraft_list(state_json):
    for key in ["aircraft", "planes", "entities"]:
        if key in state_json and isinstance(state_json[key], list):
            return state_json[key]
    return []

# Reset environment
if st.button("Reset Environment"):
    try:
        response = requests.post(f"{BASE_URL}/reset", json={"task": TASK})
        response.raise_for_status()
        st.session_state["state"] = response.json()
        st.session_state["step"] = 0
        st.success("Environment reset!")
        st.json(st.session_state["state"])  # Inspect JSON keys
    except Exception as e:
        st.error(f"Failed to reset environment: {e}")

# Next step
if st.session_state["state"] is not None:
    if st.button("Next Step"):
        try:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"task": TASK, "actions": []}  # Can later add real actions
            )
            response.raise_for_status()
            st.session_state["state"] = response.json()
            st.session_state["step"] += 1
        except Exception as e:
            st.error(f"Failed to step: {e}")

    st.write(f"**Step {st.session_state['step']}**")

    # Get aircraft/entities from state
    aircraft_list = get_aircraft_list(st.session_state["state"])
    
    if not aircraft_list:
        st.warning("No aircraft/planes found in response!")
    else:
        # Print aircraft info safely
        for ac in aircraft_list:
            st.write(
                f"Callsign: {ac.get('callsign', 'N/A')}, "
                f"X: {float(ac.get('x', 0) or 0):.2f}, "
                f"Y: {float(ac.get('y', 0) or 0):.2f}, "
                f"Alt: {float(ac.get('altitude', 0) or 0):.2f}, "
                f"Heading: {float(ac.get('heading', 0) or 0):.2f}"
            )

        # Wake turbulence check
        for i in range(len(aircraft_list)):
            for j in range(i + 1, len(aircraft_list)):
                ac1 = aircraft_list[i]
                ac2 = aircraft_list[j]
                x1, y1 = float(ac1.get("x", 0) or 0), float(ac1.get("y", 0) or 0)
                x2, y2 = float(ac2.get("x", 0) or 0), float(ac2.get("y", 0) or 0)
                dx, dy = x1 - x2, y1 - y2
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < 5:  # threshold for turbulence
                    st.warning(
                        f"⚠️ Wake turbulence risk between {ac1.get('callsign', '?')} "
                        f"and {ac2.get('callsign', '?')} (Distance: {dist:.2f})"
                    )
"""
AffectSense AI: Human Emotion & Retinal Gaze Intelligence Studio
================================================================
Clean, minimalist, professional interface for multi-modal affect analysis.
"""

import json
import os
import sys
import tempfile
import time
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, List, Optional, Tuple
import uuid

from engine.detector import SensoryPipeline, SensoryResult
from engine.auth import AuthManager
from engine.admin_view import render_admin_dashboard

# Page configuration
st.set_page_config(
    page_title="AffectSense AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist, Clean, Professional Styling (Zero Neon, Zero Flashy Effects)
st.markdown("""
<style>
    /* Clean base styling */
    html, body, [class*="css"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937 !important;
    }

    /* Simple, clean, elegant buttons (NO white background, NO neon) */
    button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"],
    .stButton > button, 
    .stDownloadButton > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stButton"] > button {
        background-color: #1e293b !important;
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        color: #f1f5f9 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1rem !important;
        box-shadow: none !important;
        text-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
        width: 100% !important;
    }

    button *,
    button p,
    button span,
    button div,
    .stButton button *,
    .stDownloadButton button * {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
        text-shadow: none !important;
    }

    button:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    .stButton > button:hover, 
    .stDownloadButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stButton"] > button:hover {
        background-color: #334155 !important;
        background: #334155 !important;
        border-color: #475569 !important;
        color: #ffffff !important;
        box-shadow: none !important;
        transform: none !important;
    }

    button:hover *,
    button:hover p,
    button:hover span {
        color: #ffffff !important;
        text-shadow: none !important;
    }

    /* Minimal Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 600;
        color: #f8fafc;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Diagnosis Box */
    .diag-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 18px;
    }
    .badge-simple {
        display: inline-block;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
    }
    .badge-mask {
        background: #451a1a;
        color: #fca5a5;
        border: 1px solid #7f1d1d;
    }
    .badge-genuine {
        background: #143522;
        color: #86efac;
        border: 1px solid #166534;
    }
    .badge-love {
        background: #401824;
        color: #fbcfe8;
        border: 1px solid #831843;
    }
    .badge-neutral {
        background: #1e293b;
        color: #93c5fd;
        border: 1px solid #2563eb;
    }

    /* Dual Justifications */
    .just-box {
        background: #182234;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .just-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .just-body {
        font-size: 0.88rem;
        color: #e2e8f0;
        line-height: 1.5;
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #1e293b;
        padding: 4px;
        border-radius: 6px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 500;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #334155 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# ACCESS KEY SECURITY GATEKEEPER
# -------------------------------------------------------------------------
@st.cache_resource
def get_auth_manager():
    return AuthManager()

auth_manager = get_auth_manager()

# Ensure unique client session ID for rate limiting
if "client_session_id" not in st.session_state:
    st.session_state.client_session_id = str(uuid.uuid4())

# Session status check
if auth_manager.auth_enabled:
    current_session = st.session_state.get("auth_session")
    is_valid, remaining_sec, session_msg = auth_manager.is_session_valid(current_session)
    
    # Check if session was previously active and just expired
    was_active = st.session_state.get("session_was_active", False)
    
    if not is_valid:
        st.session_state.auth_session = None
        st.session_state.session_was_active = False

        # Centered Dark-Mode Access Portal
        col_pad1, col_center, col_pad2 = st.columns([1, 2.0, 1])
        with col_center:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 12px; padding: 26px 24px; margin-top: 30px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);">
                <div style="text-align: center; margin-bottom: 16px;">
                    <div style="font-size: 2.3rem; margin-bottom: 6px;">🔐</div>
                    <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">AffectSense AI Access Portal</h2>
                    <p style="margin-top: 4px; font-size: 0.85rem; color: #94a3b8;">Human Emotion & Retinal Sense Intelligence Studio</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if was_active:
                st.warning("⏱️ **Session Expired**: Your session has concluded. Please enter a new Access Key to unlock the studio.")
            else:
                st.info("🛡️ **Protected Studio**: Access requires a single-use Access Key or an authorized Admin Key.")

            # Input form
            with st.form("auth_form", clear_on_submit=False):
                key_input = st.text_input(
                    "Enter Access Key",
                    value=st.session_state.get("prefill_key", ""),
                    placeholder="e.g. AFTS-XXXX-XXXX-XXXX-XXXX",
                    help="Enter your issued 16-character access key."
                )

                st.markdown("""
                <div style="margin-top: 14px; margin-bottom: 6px;">
                    <span style="font-size: 0.84rem; font-weight: 600; color: #cbd5e1;">User Verification Identity</span>
                    <span style="font-size: 0.74rem; color: #94a3b8; margin-left: 6px;">(Required for users; optional for admins)</span>
                </div>
                """, unsafe_allow_html=True)
                
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    name_input = st.text_input("Full Name", placeholder="e.g. Jane Doe", help="Required for regular users")
                with col_u2:
                    phone_input = st.text_input("Phone Number", placeholder="e.g. +1 555 123 4567", help="Required for regular users")
                
                email_input = st.text_input("Email Address", placeholder="e.g. jane@example.com", help="Required for regular users")
                
                st.caption("ℹ️ **Admins**: Contact fields are optional when using an Admin Key. **Users**: Name, Phone, and Email are required.")
                
                btn_unlock = st.form_submit_button("🔓 Unlock AffectSense Studio", use_container_width=True)

            if btn_unlock:
                if not key_input.strip():
                    st.error("Please enter an Access Key.")
                else:
                    user_payload = {
                        "name": name_input.strip(),
                        "phone": phone_input.strip(),
                        "email": email_input.strip(),
                    }
                    ok, msg, new_session = auth_manager.validate_and_activate(
                        key_input.strip(),
                        user_info=user_payload,
                        client_info={"session_id": st.session_state.client_session_id}
                    )
                    if ok:
                        st.session_state.auth_session = new_session
                        st.session_state.session_was_active = True
                        st.session_state.pop("prefill_key", None)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("""
            <div style="background: #182234; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-top: 14px;">
                <div style="font-size: 0.78rem; color: #94a3b8;">
                    🔒 <b>Need an Access Key?</b><br>
                    Contact the system owner or administrator to request an Access Key.<br>
                    If you are the repository owner, run <code>python make_key.py</code> in your local terminal.
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("ℹ️ How Access Keys Work"):
                st.markdown(f"""
                - **Centralized Tracking**: All key lifecycles, user verifications, and sessions are tracked securely in MongoDB.
                - **Single-Use User Keys**: Each regular key can only be redeemed once and unlocks the studio for forensic analysis.
                - **Identity Verification**: Regular users verify their Name, Phone, and Email upon redemption.
                - **Multiple Admin Keys**: Admins bypass user requirements and gain access to the central Admin Dashboard.
                - **Local Generation**: Keys can only be minted on the owner's private local machine.
                """)

        # HALT execution here: no tabs, camera, or pipeline exposed
        st.stop()
    else:
        st.session_state.session_was_active = True
        auth_manager.set_current_session(current_session)

# Determine user and admin status
is_master = current_session.get("is_master", False) if current_session else False
is_admin = current_session.get("is_admin", False) or is_master if current_session else False
user_info = current_session.get("user_info", {}) if current_session else {}
user_name = user_info.get("name") or ("Admin" if is_admin else "Authorized User")

# Sidebar
admin_view_choice = "👁️ AffectSense Studio"
with st.sidebar:
    # Session Timer Badge
    is_valid, remaining_sec, _ = auth_manager.is_session_valid(current_session)
    if not is_valid:
        st.session_state.auth_session = None
        st.session_state.session_was_active = True
        st.rerun()

    time_str = auth_manager.format_time_remaining(remaining_sec)
    key_disp = current_session.get("key_display", "ACTIVE-KEY") if current_session else "AUTH-OK"
    
    status_label = "ADMIN ACCESS" if is_admin else f"⏱️ {time_str}"
    badge_color = "#38bdf8" if (is_admin or remaining_sec > 60) else "#f87171"
    
    st.markdown(f"""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Active Session</span>
            <span style="font-size: 0.72rem; color: #64748b; font-family: monospace;">{key_disp}</span>
        </div>
        <div style="font-size: 1.25rem; font-weight: 700; color: {badge_color}; font-family: monospace; margin-top: 4px;">
            {status_label}
        </div>
        <div style="font-size: 0.76rem; color: #cbd5e1; margin-top: 4px;">
            👤 {user_name}
        </div>
        <div style="font-size: 0.70rem; color: #64748b; margin-top: 2px;">
            {'Multi-Session Admin' if is_admin else f"Window: {int(current_session.get('duration_minutes', 10))} mins"}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔒 Lock", key="btn_lock_session"):
            st.session_state.auth_session = None
            st.session_state.session_was_active = False
            st.rerun()
    with col_sb2:
        if st.button("🔄 Check", key="btn_refresh_timer"):
            st.rerun()
            
    st.markdown("---")

    # Navigation Switcher for Admins
    if is_admin:
        st.markdown("<p style='font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>Admin Navigation</p>", unsafe_allow_html=True)
        admin_view_choice = st.radio(
            "Admin Navigation",
            ["👁️ AffectSense Studio", "🛡️ Admin Key Tracker"],
            index=st.session_state.get("admin_nav_index", 0),
            key="admin_nav_radio",
            label_visibility="collapsed"
        )
        st.session_state.admin_nav_index = 0 if admin_view_choice == "👁️ AffectSense Studio" else 1
        st.markdown("---")

    if admin_view_choice == "👁️ AffectSense Studio":
        st.markdown("### AffectSense")
        st.caption("Human Emotion & Gaze Sense Identification")
        st.markdown("---")

        st.markdown("**Overview**")
        st.markdown("""
        - Retinal Gaze (Pitch & Yaw)
        - Duchenne Eye-Mouth Coherence
        - 25+ Emotional States
        - Involuntary Micro-Expressions
        - Contactless Pulse (rPPG)
        """)
        st.markdown("---")

        st.markdown("**Options**")
        show_mesh = st.checkbox("Show Face Mesh", value=True)
        show_gaze_rays = st.checkbox("Show Gaze Vectors", value=True)

# Main Screen Routing: If Admin selected Key Tracker, render and halt
if is_admin and admin_view_choice == "🛡️ Admin Key Tracker":
    render_admin_dashboard(auth_manager)
    st.stop()

@st.cache_resource
def get_pipeline():
    import importlib
    import engine.detector
    importlib.reload(engine.detector)
    return engine.detector.SensoryPipeline()


pipeline = get_pipeline()
if not hasattr(pipeline, "reset_tracker") or not hasattr(pipeline, "process_frame_multi"):
    st.cache_resource.clear()
    pipeline = get_pipeline()

# Main Header
st.markdown("## Human Emotion & Retinal Sense Detector")
st.caption("Identify true underlying human affect through eye gaze orientation, micro-expressions, and Duchenne coherence.")
st.markdown("---")



def render_single_person_details(result: SensoryResult, prefix_key: str = ""):
    s_score = result.affect.sincerity_score
    is_masking = result.affect.is_masking_detected
    is_love = "Love" in result.affect.primary_state or "Affection" in result.affect.primary_state

    col_d1, col_d2 = st.columns([1.1, 1.0])
    with col_d1:
        st.markdown('<div class="diag-box">', unsafe_allow_html=True)
        if is_love:
            st.markdown('<div class="badge-simple badge-love">ROMANTIC AFFECTION / IN LOVE</div>', unsafe_allow_html=True)
        elif is_masking:
            st.markdown('<div class="badge-simple badge-mask">MASKED / INCONGRUENT EMOTION</div>', unsafe_allow_html=True)
        elif s_score >= 0.70:
            st.markdown('<div class="badge-simple badge-genuine">AUTHENTIC / CONGRUENT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-simple badge-neutral">NEUTRAL / INWARD</div>', unsafe_allow_html=True)

        st.markdown(f"<h3 style='margin: 4px 0 8px 0; color: #f8fafc;'>Person {result.track_id}: {result.affect.primary_state}</h3>", unsafe_allow_html=True)
        st.markdown(f"**Apparent Expression:** `{result.affect.surface_expression}`")
        st.markdown(f"**Underlying Truth:** `{result.affect.underlying_truth}`")

        bar_color = "#38bdf8" if is_love else ("#ef4444" if s_score < 0.45 else ("#22c55e" if s_score >= 0.70 else "#f59e0b"))
        st.markdown(f"""
        <div style="margin-top: 14px; margin-bottom: 4px; display: flex; justify-content: space-between; font-size: 0.84rem;">
            <span style="color: #94a3b8;">Sincerity Level</span>
            <span style="font-weight: 600; color: {bar_color};">{int(s_score * 100)}%</span>
        </div>
        <div style="width: 100%; background: #0f172a; height: 8px; border-radius: 4px; overflow: hidden; border: 1px solid #334155;">
            <div style="width: {int(s_score * 100)}%; background: {bar_color}; height: 100%;"></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d2:
        st.markdown(f"""
        <div class="just-box">
            <div class="just-title">Scientific Justification</div>
            <div class="just-body">{result.affect.scientific_justification}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="just-box">
            <div class="just-title">Plain English Explanation</div>
            <div class="just-body">{result.affect.layman_justification}</div>
        </div>
        """, unsafe_allow_html=True)

    # Biometric Measurements Grid
    st.markdown("##### Detailed Biometric Measurements")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Retina Pitch (Vertical)</div>
            <div class="metric-value">{result.gaze.avg_pitch:+.1f}°</div>
            <div class="metric-sub">{result.gaze.gaze_direction}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Retina Yaw (Horizontal)</div>
            <div class="metric-value">{result.gaze.avg_yaw:+.1f}°</div>
            <div class="metric-sub">Horizontal Angle</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Smile (AU12)</div>
            <div class="metric-value">{result.aus.au12_lip_corner_puller:.2f}</div>
            <div class="metric-sub">Lip Corner Pull</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Cheek Raiser (AU6)</div>
            <div class="metric-value">{result.aus.au6_cheek_raiser:.2f}</div>
            <div class="metric-sub">Eye Crinkle</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duchenne Coherence</div>
            <div class="metric-value">{result.aus.duchenne_coherence:.2f}</div>
            <div class="metric-sub">Eye-Mouth Harmony</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Grief Brow (AU1)</div>
            <div class="metric-value">{result.aus.au1_inner_brow_raiser:.2f}</div>
            <div class="metric-sub">Inner Brow Elevation</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Eye Aperture (EAR)</div>
            <div class="metric-value">{result.gaze.avg_ear:.3f}</div>
            <div class="metric-sub">Openness Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Heart Rate (rPPG)</div>
            <div class="metric-value">{result.pulse.bpm:.0f} BPM</div>
            <div class="metric-sub">Contactless Pulse</div>
        </div>
        """, unsafe_allow_html=True)

    # Radar & Action Units Charts
    st.markdown("---")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("##### Emotion Probability Radar")
        radar_data = result.affect.emotion_radar
        categories = list(radar_data.keys())
        values = list(radar_data.values())
        categories.append(categories[0])
        values.append(values[0])

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.15)',
            line=dict(color='#38bdf8', width=1.5),
            marker=dict(size=4, color='#38bdf8')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.0], gridcolor='#334155'),
                angularaxis=dict(gridcolor='#334155', linecolor='#334155')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', size=10),
            height=320,
            margin=dict(l=35, r=35, t=25, b=25)
        )
        st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{prefix_key}_{result.track_id}")

    with r_col2:
        st.markdown("##### Action Unit Intensities")
        au_dict = {
            "Smile (AU12)": result.aus.au12_lip_corner_puller,
            "Cheek Raiser (AU6)": result.aus.au6_cheek_raiser,
            "Squint (AU7)": result.aus.au7_lid_tightener,
            "Grief Brow (AU1)": result.aus.au1_inner_brow_raiser,
            "Brow Furrow (AU4)": result.aus.au4_brow_lowerer,
            "Frown (AU15)": result.aus.au15_lip_corner_depressor,
            "Lip Press (AU24)": result.aus.au24_lip_pressor,
            "Wide Eye (AU5)": result.aus.au5_upper_lid_raiser
        }
        df_au = pd.DataFrame({"Action Unit": list(au_dict.keys()), "Intensity": list(au_dict.values())})
        fig_bar = px.bar(
            df_au,
            x="Intensity",
            y="Action Unit",
            orientation="h",
            color="Intensity",
            color_continuous_scale=["#1e293b", "#334155", "#38bdf8"],
            range_x=[0, 1.0]
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', size=10),
            height=320,
            margin=dict(l=10, r=10, t=25, b=25),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{prefix_key}_{result.track_id}")

    # Export individual JSON button
    json_data = {
        "person_id": f"Person {result.track_id}",
        "primary_state": result.affect.primary_state,
        "sincerity_score": result.affect.sincerity_score,
        "is_masking_detected": result.affect.is_masking_detected,
        "scientific_justification": result.affect.scientific_justification,
        "layman_justification": result.affect.layman_justification,
        "gaze": {
            "pitch_deg": result.gaze.avg_pitch,
            "yaw_deg": result.gaze.avg_yaw,
            "direction": result.gaze.gaze_direction,
            "ear": result.gaze.avg_ear
        },
        "action_units": {
            "au12_smile": result.aus.au12_lip_corner_puller,
            "au6_cheek": result.aus.au6_cheek_raiser,
            "au1_grief": result.aus.au1_inner_brow_raiser,
            "au4_furrow": result.aus.au4_brow_lowerer,
            "duchenne_coherence": result.aus.duchenne_coherence
        },
        "pulse_bpm": result.pulse.bpm,
        "diagnostic_notes": result.affect.diagnostic_notes,
        "emotion_radar": result.affect.emotion_radar
    }
    st.download_button(
        label=f"Export Person {result.track_id} JSON Telemetry",
        data=json.dumps(json_data, indent=2),
        file_name=f"affectsense_person_{result.track_id}.json",
        mime="application/json",
        key=f"dl_{prefix_key}_{result.track_id}"
    )


def render_multi_affect_dashboard(results: List[SensoryResult], original_img: np.ndarray):
    """Renders comprehensive multi-person dashboard with individual deep dives and comparison."""
    annotated = pipeline.draw_hud_multi(original_img, results, show_mesh=show_mesh, show_gaze_rays=show_gaze_rays)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    num_f = len(results)
    st.markdown(f"##### Annotated Photo Analysis — **{num_f} Face{'s' if num_f > 1 else ''} Detected & Tracked**")
    col_img, col_quick = st.columns([1.25, 1.0])

    with col_img:
        st.image(annotated_rgb, use_container_width=True)
        _, buf = cv2.imencode(".jpg", annotated)
        st.download_button(
            label="Download Analyzed Image (All Annotations)",
            data=buf.tobytes(),
            file_name="affectsense_multiface_analysis.jpg",
            mime="image/jpeg",
            key="dl_annotated_img"
        )

    with col_quick:
        st.markdown('<div class="diag-box">', unsafe_allow_html=True)
        st.markdown(f"<h4 style='margin:0 0 10px 0; color:#f8fafc;'>Multi-Face Overview ({num_f} People)</h4>", unsafe_allow_html=True)
        for res in results:
            s_score = res.affect.sincerity_score
            is_m = res.affect.is_masking_detected
            color = "#fca5a5" if is_m else ("#86efac" if s_score >= 0.70 else "#93c5fd")
            st.markdown(f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #f1f5f9; font-size: 0.95rem;">👤 Person {res.track_id}</strong>
                    <span style="font-size: 0.78rem; font-weight: 600; color: {color}; border: 1px solid {color}44; padding: 2px 6px; border-radius: 4px;">{res.affect.primary_state}</span>
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 6px;">
                    Sincerity: <strong style="color: {color};">{int(s_score * 100)}%</strong> | 
                    Gaze: {res.gaze.avg_pitch:+.1f}° ({res.gaze.gaze_direction}) | 
                    Smile: {res.aus.au12_lip_corner_puller:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    if num_f == 1:
        st.markdown("#### Individual Analysis: Person 1")
        render_single_person_details(results[0], prefix_key="single_main")
    else:
        st.markdown("#### 🔬 Individual Face Forensics: Select Person to Inspect")
        tab_titles = ["👥 Side-by-Side Comparison"] + [f"👤 Person {r.track_id}: {r.affect.primary_state[:16]}" for r in results]
        face_tabs = st.tabs(tab_titles)

        with face_tabs[0]:
            st.markdown("##### Direct Side-by-Side Biometric Comparison")
            comp_rows = []
            for r in results:
                comp_rows.append({
                    "Person": f"Person {r.track_id}",
                    "Primary Feeling": r.affect.primary_state,
                    "Sincerity Index": f"{int(r.affect.sincerity_score * 100)}%",
                    "Masking Detected": "⚠️ YES" if r.affect.is_masking_detected else "✅ NO",
                    "Retina Pitch": f"{r.gaze.avg_pitch:+.1f}°",
                    "Gaze Direction": r.gaze.gaze_direction,
                    "Smile (AU12)": f"{r.aus.au12_lip_corner_puller:.2f}",
                    "Cheek (AU6)": f"{r.aus.au6_cheek_raiser:.2f}",
                    "Duchenne Coherence": f"{r.aus.duchenne_coherence:.2f}",
                    "Pulse (rPPG)": f"{r.pulse.bpm:.0f} BPM"
                })
            df_comp = pd.DataFrame(comp_rows)
            st.dataframe(df_comp, use_container_width=True)

            # Dyadic Emotional Dynamics summary
            p1_state = results[0].affect.primary_state
            p2_state = results[1].affect.primary_state
            is_congruent = (results[0].affect.is_masking_detected == results[1].affect.is_masking_detected)
            st.markdown(f"""
            <div class="just-box" style="margin-top: 12px;">
                <div class="just-title">Interpersonal Emotional Dynamics</div>
                <div class="just-body">
                    <strong>Person 1:</strong> {p1_state} ({int(results[0].affect.sincerity_score*100)}% Sincerity)<br>
                    <strong>Person 2:</strong> {p2_state} ({int(results[1].affect.sincerity_score*100)}% Sincerity)<br><br>
                    {'✅ <strong>Emotional Congruence:</strong> Both participants exhibit emotionally synchronized affective baseline responses.' if is_congruent else '⚠️ <strong>Affective Divergence:</strong> One person is exhibiting masked/suppressed affect while the other is displaying open emotional engagement.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Combined JSON export
            all_json = [
                {
                    "person_id": f"Person {r.track_id}",
                    "primary_state": r.affect.primary_state,
                    "sincerity_score": r.affect.sincerity_score,
                    "is_masking": r.affect.is_masking_detected,
                    "gaze_pitch": r.gaze.avg_pitch,
                    "gaze_yaw": r.gaze.avg_yaw,
                    "gaze_direction": r.gaze.gaze_direction,
                    "action_units": {
                        "au12_smile": r.aus.au12_lip_corner_puller,
                        "au6_cheek": r.aus.au6_cheek_raiser,
                        "au1_grief": r.aus.au1_inner_brow_raiser,
                        "duchenne_coherence": r.aus.duchenne_coherence
                    },
                    "scientific_justification": r.affect.scientific_justification,
                    "layman_justification": r.affect.layman_justification
                }
                for r in results
            ]
            st.download_button(
                label="Download Complete Multi-Person JSON Telemetry",
                data=json.dumps(all_json, indent=2),
                file_name="affectsense_all_people_telemetry.json",
                mime="application/json",
                key="dl_all_json_comp"
            )

        for idx, r in enumerate(results):
            with face_tabs[idx + 1]:
                render_single_person_details(r, prefix_key=f"tab_p{r.track_id}")


def render_affect_dashboard(result: SensoryResult, original_img: np.ndarray):
    """Backward-compatible wrapper for single-result dashboard callers."""
    render_multi_affect_dashboard([result], original_img)


# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
tab_inspect, tab_live_rec, tab_video, tab_science = st.tabs([
    "Photo Inspector",
    "Live Video & Mic Recording",
    "Video Timeline Analyzer",
    "Emotion Catalog & Science"
])

# =========================================================================
# TAB 1: PHOTO INSPECTOR
# =========================================================================
with tab_inspect:
    st.markdown("##### Analyze Portrait or Group Photo")
    st.write("Select a test benchmark or upload an image (detects single or multiple faces):")

    p1, p2, p3, p4, p5 = st.columns(5)

    selected_sample = None
    with p1:
        if st.button("Masked Sadness\n(Fake Smile)"):
            selected_sample = "sample_data/masked_sadness.jpg"
    with p2:
        if st.button("Genuine Joy\n(Duchenne Smile)"):
            selected_sample = "sample_data/genuine_joy.jpg"
    with p3:
        if st.button("In Love\n(Affection)"):
            selected_sample = "sample_data/in_love.jpg"
    with p4:
        if st.button("Inward Focus\n(Neutral)"):
            selected_sample = "sample_data/neutral_focus.jpg"
    with p5:
        if st.button("Dual Face\n(Masked vs Joy)"):
            selected_sample = "dual_benchmark"

    uploaded_file = st.file_uploader("Or upload image (PNG, JPG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

    target_img = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        target_img = cv2.imdecode(file_bytes, 1)
    elif selected_sample == "dual_benchmark":
        img1 = cv2.imread("sample_data/masked_sadness.jpg")
        img2 = cv2.imread("sample_data/genuine_joy.jpg")
        if img1 is not None and img2 is not None:
            h = min(img1.shape[0], img2.shape[0])
            img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
            img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
            target_img = np.hstack([img1, img2])
    elif selected_sample is not None and os.path.exists(selected_sample):
        target_img = cv2.imread(selected_sample)

    if target_img is not None:
        with st.spinner("Detecting and analyzing all faces in image..."):
            multi_res = pipeline.process_frame_multi(target_img, is_static=True)

        if multi_res and len(multi_res) > 0:
            render_multi_affect_dashboard(multi_res, target_img)
        else:
            st.error("No face detected in the image.")
    else:
        st.info("Click one of the benchmark buttons above or upload an image to begin.")


# =========================================================================
# TAB 2: LIVE VIDEO STREAMING & AUDIO/MIC RECORDER
# =========================================================================
with tab_live_rec:
    st.markdown("##### Live Camera Stream & Microphone Audio Recording")
    st.write("Start your webcam to see real-time floating context overlays, and record video with microphone audio:")

    # Real-Time MediaPipe Face Tracking & Audio/Video Recorder
    html5_recorder_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <!-- MediaPipe FaceMesh & Camera Utils -->
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js" crossorigin="anonymous"></script>
        <style>
            body {
                background: #0f172a;
                color: #e2e8f0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 0;
                padding: 6px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .video-container {
                position: relative;
                width: 100%;
                max-width: 820px;
                aspect-ratio: 16 / 9;
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                overflow: hidden;
            }
            video {
                width: 100%;
                height: 100%;
                object-fit: cover;
                transform: scaleX(-1);
            }
            canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                transform: scaleX(-1);
            }
            .hud-controls {
                display: flex;
                gap: 10px;
                margin-top: 12px;
                flex-wrap: wrap;
                justify-content: center;
            }
            .btn {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #f1f5f9;
                font-weight: 500;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.88rem;
                transition: background-color 0.15s ease;
            }
            .btn:hover {
                background-color: #334155;
                color: #ffffff;
            }
            .btn-rec {
                background-color: #3b1818;
                border-color: #7f1d1d;
                color: #fca5a5;
            }
            .btn-rec:hover {
                background-color: #501d1d;
                color: #ffffff;
            }
            .btn-stop {
                background-color: #1e293b;
                border-color: #475569;
                color: #94a3b8;
            }
            .rec-badge {
                position: absolute;
                top: 12px;
                left: 12px;
                background: rgba(185, 28, 28, 0.9);
                color: #fff;
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 0.78rem;
                font-weight: 600;
                display: none;
                align-items: center;
                gap: 6px;
                z-index: 10;
            }
            .rec-dot {
                width: 8px;
                height: 8px;
                background: #fff;
                border-radius: 50%;
                animation: blink 1s infinite;
            }
            @keyframes blink {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
            .playback-box {
                margin-top: 14px;
                width: 100%;
                max-width: 820px;
                display: none;
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 12px;
            }
            #status-bar {
                margin-top: 8px;
                font-size: 0.82rem;
                color: #94a3b8;
            }
        </style>
    </head>
    <body>
        <div class="video-container">
            <video id="webcam" autoplay playsinline muted></video>
            <canvas id="ar-canvas"></canvas>
            <div id="recBadge" class="rec-badge">
                <div class="rec-dot"></div>
                <span id="recTimer">RECORDING (VIDEO + MIC) 00:00</span>
            </div>
        </div>

        <div id="status-bar">Status: Click 'Turn On Camera' to start real-time biometric face detection</div>

        <div class="hud-controls">
            <button id="btnStartCam" class="btn" onclick="startCamera()">📷 Turn On Camera & Detector</button>
            <button id="btnStartRec" class="btn btn-rec" onclick="startRecording()" disabled>🔴 Record (Video + Voice Mic)</button>
            <button id="btnStopRec" class="btn btn-stop" onclick="stopRecording()" disabled>⏹️ Stop Recording</button>
        </div>

        <div id="playbackContainer" class="playback-box">
            <div style="font-size: 0.88rem; font-weight: 600; margin-bottom: 8px; color: #f1f5f9;">Recorded Video with Microphone Sound:</div>
            <video id="playbackVideo" controls style="width: 100%; border-radius: 6px;"></video>
            <div style="margin-top: 10px;">
                <a id="downloadLink" class="btn" style="text-decoration: none; display: inline-block;">⬇️ Download Video (.webm)</a>
            </div>
        </div>

        <script>
            let stream = null;
            let camera = null;
            let faceMesh = null;
            let mediaRecorder = null;
            let recordedChunks = [];
            let timerInterval = null;
            let secondsRecorded = 0;

            const video = document.getElementById('webcam');
            const canvas = document.getElementById('ar-canvas');
            const ctx = canvas.getContext('2d');
            const statusBar = document.getElementById('status-bar');
            const recBadge = document.getElementById('recBadge');
            const recTimer = document.getElementById('recTimer');
            const btnStartCam = document.getElementById('btnStartCam');
            const btnStartRec = document.getElementById('btnStartRec');
            const btnStopRec = document.getElementById('btnStopRec');
            const playbackContainer = document.getElementById('playbackContainer');
            const playbackVideo = document.getElementById('playbackVideo');
            const downloadLink = document.getElementById('downloadLink');

            async function startCamera() {
                try {
                    statusBar.innerText = "Initializing Camera & MediaPipe Face Mesh neural models...";
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: 1280, height: 720, facingMode: "user" },
                        audio: true
                    });
                    video.srcObject = stream;
                    btnStartCam.disabled = true;
                    btnStartCam.innerText = "✅ Camera & Detector Active";
                    btnStartRec.disabled = false;

                    video.onloadedmetadata = () => {
                        canvas.width = video.videoWidth || 1280;
                        canvas.height = video.videoHeight || 720;
                    };

                    initMediaPipe();
                } catch (err) {
                    statusBar.innerText = "Camera / Mic error: " + err.message;
                    alert("Camera/Mic Error: " + err.message);
                }
            }

            function initMediaPipe() {
                faceMesh = new FaceMesh({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
                });

                faceMesh.setOptions({
                    maxNumFaces: 4,
                    refineLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                });

                faceMesh.onResults(onFaceResults);

                camera = new Camera(video, {
                    onFrame: async () => {
                        await faceMesh.send({ image: video });
                    },
                    width: 1280,
                    height: 720
                });
                camera.start();
                statusBar.innerText = "Real-Time Face Detector Running: Point camera toward faces";
            }

            function dist(p1, p2, w, h) {
                return Math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h);
            }

            // Persistent multi-face tracker across camera frames
            let liveTracks = []; // { id: number, cx: number, cy: number, unseen: number }
            let nextLiveId = 1;

            function onFaceResults(results) {
                const w = canvas.width;
                const h = canvas.height;
                ctx.save();
                ctx.clearRect(0, 0, w, h);

                if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                    statusBar.innerText = "Status: Searching for faces in camera stream...";
                    for (let t of liveTracks) { t.unseen++; }
                    liveTracks = liveTracks.filter(t => t.unseen < 20);
                    ctx.restore();
                    return;
                }

                const numFaces = results.multiFaceLandmarks.length;
                statusBar.innerText = `Status: ${numFaces} Face(s) Detected & Tracked simultaneously (Persistent Person IDs)`;

                // 1. Extract bounding boxes & centroids for all detected faces
                const detectedList = [];
                for (let f = 0; f < numFaces; f++) {
                    const lm = results.multiFaceLandmarks[f];
                    let minX = w, maxX = 0, minY = h, maxY = 0;
                    for (let i = 0; i < lm.length; i++) {
                        const px = lm[i].x * w;
                        const py = lm[i].y * h;
                        if (px < minX) minX = px;
                        if (px > maxX) maxX = px;
                        if (py < minY) minY = py;
                        if (py > maxY) maxY = py;
                    }
                    const padX = (maxX - minX) * 0.08;
                    const padY = (maxY - minY) * 0.08;
                    minX = Math.max(0, minX - padX);
                    maxX = Math.min(w, maxX + padX);
                    minY = Math.max(0, minY - padY);
                    maxY = Math.min(h, maxY + padY);
                    detectedList.push({
                        lm: lm,
                        minX: minX, maxX: maxX, minY: minY, maxY: maxY,
                        fw: maxX - minX, fh: maxY - minY,
                        cx: (minX + maxX) / 2, cy: (minY + maxY) / 2
                    });
                }

                // 2. Multi-face track association using centroid distance
                for (let t of liveTracks) { t.unseen++; }

                const matchedDet = new Set();
                const matchedTracks = new Set();
                const assignments = [];

                const pairs = [];
                for (let d = 0; d < detectedList.length; d++) {
                    for (let t = 0; t < liveTracks.length; t++) {
                        const dDist = Math.hypot(detectedList[d].cx - liveTracks[t].cx, detectedList[d].cy - liveTracks[t].cy);
                        if (dDist < w * 0.35) {
                            pairs.push({ d: d, t: t, dist: dDist });
                        }
                    }
                }
                pairs.sort((a, b) => a.dist - b.dist);

                for (let p of pairs) {
                    if (!matchedDet.has(p.d) && !matchedTracks.has(p.t)) {
                        matchedDet.add(p.d);
                        matchedTracks.add(p.t);
                        const trk = liveTracks[p.t];
                        trk.cx = detectedList[p.d].cx;
                        trk.cy = detectedList[p.d].cy;
                        trk.unseen = 0;
                        assignments.push({ trackId: trk.id, det: detectedList[p.d] });
                    }
                }

                for (let d = 0; d < detectedList.length; d++) {
                    if (!matchedDet.has(d)) {
                        const newId = nextLiveId++;
                        liveTracks.push({ id: newId, cx: detectedList[d].cx, cy: detectedList[d].cy, unseen: 0 });
                        assignments.push({ trackId: newId, det: detectedList[d] });
                    }
                }
                liveTracks = liveTracks.filter(t => t.unseen < 20);

                // Sort assignments by trackId for consistent drawing
                assignments.sort((a, b) => a.trackId - b.trackId);

                // 3. Render AR HUD overlays for each tracked individual
                for (let item of assignments) {
                    const pid = item.trackId;
                    const det = item.det;
                    const lm = det.lm;
                    const minX = det.minX;
                    const maxX = det.maxX;
                    const minY = det.minY;
                    const maxY = det.maxY;
                    const fw = det.fw;
                    const cx = det.cx;

                    // Eye Aspect Ratio (EAR)
                    const leftEar = dist(lm[160], lm[144], w, h) / (2 * Math.max(1, dist(lm[33], lm[133], w, h)));
                    const rightEar = dist(lm[385], lm[380], w, h) / (2 * Math.max(1, dist(lm[362], lm[263], w, h)));
                    const avgEar = (leftEar + rightEar) / 2;

                    // Smile Ratio (AU12)
                    const mouthWidth = dist(lm[61], lm[291], w, h);
                    const smileScore = mouthWidth / (fw * 0.44);

                    // Vertical Iris Gaze (Pitch)
                    const leftMidY = (lm[160].y + lm[144].y) / 2;
                    const rightMidY = (lm[385].y + lm[380].y) / 2;
                    const leftIrisY = lm[468] ? lm[468].y : leftMidY;
                    const rightIrisY = lm[473] ? lm[473].y : rightMidY;
                    const isGazeDown = ((leftIrisY - leftMidY) + (rightIrisY - rightMidY)) / 2 > 0.007;

                    // Affect Diagnosis
                    let state = "NEUTRAL FOCUS";
                    let sincerity = 0.90;
                    let badgeColor = "#2563eb";

                    if (smileScore > 1.06) {
                        if (isGazeDown && avgEar > 0.20) {
                            state = "MASKED SADNESS";
                            sincerity = 0.28;
                            badgeColor = "#b91c1c";
                        } else if (avgEar < 0.18) {
                            state = "AUTHENTIC JOY";
                            sincerity = 0.95;
                            badgeColor = "#16a34a";
                        } else {
                            state = "POLITE SMILE";
                            sincerity = 0.52;
                            badgeColor = "#d97706";
                        }
                    } else if (isGazeDown) {
                        state = "DOWNWARD GAZE";
                        sincerity = 0.76;
                        badgeColor = "#475569";
                    } else if (smileScore > 0.96 && avgEar >= 0.18 && !isGazeDown) {
                        state = "IN LOVE / WARM";
                        sincerity = 0.96;
                        badgeColor = "#db2777";
                    }

                    // DRAW FLOATING CORNER BRACKETS AROUND THIS PERSON'S FACE
                    ctx.strokeStyle = badgeColor;
                    ctx.lineWidth = 2;
                    const bl = Math.min(24, fw * 0.15);
                    // Top-left
                    ctx.beginPath(); ctx.moveTo(minX, minY + bl); ctx.lineTo(minX, minY); ctx.lineTo(minX + bl, minY); ctx.stroke();
                    // Top-right
                    ctx.beginPath(); ctx.moveTo(maxX - bl, minY); ctx.lineTo(maxX, minY); ctx.lineTo(maxX, minY + bl); ctx.stroke();
                    // Bottom-left
                    ctx.beginPath(); ctx.moveTo(minX, maxY - bl); ctx.lineTo(minX, maxY); ctx.lineTo(minX + bl, maxY); ctx.stroke();
                    // Bottom-right
                    ctx.beginPath(); ctx.moveTo(maxX - bl, maxY); ctx.lineTo(maxX, maxY); ctx.lineTo(maxX, maxY - bl); ctx.stroke();

                    // Iris tracking dots
                    if (lm[468]) {
                        ctx.fillStyle = "#38bdf8";
                        ctx.beginPath();
                        ctx.arc(lm[468].x * w, lm[468].y * h, 3, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                    if (lm[473]) {
                        ctx.fillStyle = "#38bdf8";
                        ctx.beginPath();
                        ctx.arc(lm[473].x * w, lm[473].y * h, 3, 0, 2 * Math.PI);
                        ctx.fill();
                    }

                    // FLOATING DIAGNOSIS BADGE WITH PERSON ID (Tracks face)
                    const labelText = `Person ${pid}: ${state}`;
                    const badgeY = Math.max(32, minY - 24);
                    ctx.font = "bold 12px -apple-system, BlinkMacSystemFont, sans-serif";
                    const tw = ctx.measureText(labelText).width;

                    ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
                    ctx.strokeStyle = badgeColor;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.roundRect(cx - tw/2 - 12, badgeY - 18, tw + 24, 26, 4);
                    ctx.fill(); ctx.stroke();

                    // Forehead connector line (landmark 10)
                    ctx.beginPath();
                    ctx.moveTo(lm[10].x * w, lm[10].y * h);
                    ctx.lineTo(cx, badgeY + 8);
                    ctx.strokeStyle = badgeColor;
                    ctx.stroke();

                    // Badge text (un-mirror)
                    ctx.save();
                    ctx.translate(cx, badgeY);
                    ctx.scale(-1, 1);
                    ctx.fillStyle = "#ffffff";
                    ctx.textAlign = "center";
                    ctx.fillText(labelText, 0, -4);
                    ctx.restore();

                    // Telemetry Card underneath face
                    const cardWidth = Math.max(180, fw);
                    const infoY = Math.min(h - 45, maxY + 14);
                    ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
                    ctx.strokeStyle = "#334155";
                    ctx.beginPath();
                    ctx.roundRect(cx - cardWidth / 2, infoY, cardWidth, 32, 4);
                    ctx.fill(); ctx.stroke();

                    ctx.save();
                    ctx.translate(cx, infoY + 16);
                    ctx.scale(-1, 1);
                    ctx.fillStyle = "#cbd5e1";
                    ctx.font = "11px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText(`P${pid} | SINCERITY: ${Math.round(sincerity * 100)}% | EAR: ${avgEar.toFixed(2)} | SMILE: ${smileScore > 1.05 ? 'YES' : 'NO'}`, 0, 0);
                    ctx.restore();
                }

                ctx.restore();
            }

            function startRecording() {
                recordedChunks = [];
                const options = { mimeType: 'video/webm;codecs=vp8,opus' };
                try {
                    mediaRecorder = new MediaRecorder(stream, options);
                } catch (e) {
                    mediaRecorder = new MediaRecorder(stream);
                }

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    const blob = new Blob(recordedChunks, { type: 'video/webm' });
                    const url = URL.createObjectURL(blob);
                    playbackVideo.src = url;
                    downloadLink.href = url;
                    downloadLink.download = "affectsense_recording_" + Date.now() + ".webm";
                    playbackContainer.style.display = "block";
                };

                mediaRecorder.start(250);
                btnStartRec.disabled = true;
                btnStopRec.disabled = false;
                recBadge.style.display = "flex";

                secondsRecorded = 0;
                timerInterval = setInterval(() => {
                    secondsRecorded++;
                    const m = String(Math.floor(secondsRecorded / 60)).padStart(2, '0');
                    const s = String(secondsRecorded % 60).padStart(2, '0');
                    recTimer.innerText = `RECORDING (VIDEO + MIC) ${m}:${s}`;
                }, 1000);
            }

            function stopRecording() {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                clearInterval(timerInterval);
                recBadge.style.display = "none";
                btnStartRec.disabled = false;
                btnStopRec.disabled = true;
            }
        </script>
    </body>
    </html>
    """
    components.html(html5_recorder_code, height=640)


    st.markdown("---")
    st.markdown("##### Camera Frame Snapshot")
    cam_shot = st.camera_input("Take Snapshot for Analysis")
    if cam_shot is not None:
        bytes_data = cam_shot.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        with st.spinner("Analyzing frame..."):
            res = pipeline.process_frame(cv2_img)
        if res:
            render_affect_dashboard(res, cv2_img)
        else:
            st.warning("No face detected.")


# =========================================================================
# TAB 3: VIDEO TIMELINE ANALYZER (LIVE SCAN & PLAYBACK)
# =========================================================================
with tab_video:
    st.markdown("##### Video Emotional Sincerity Scanner & Timeline")
    st.write(
        "Upload any video clip to **play the video live while scanning facial expressions**, "
        "tracking retinal gaze vectors, and identifying true underlying feelings across the entire timeline."
    )

    vid_file = st.file_uploader("Upload Video File (MP4 / WebM / MOV / AVI)", type=["mp4", "webm", "mov", "avi"])

    if vid_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        st.write(f"Video loaded: **{total_frames} frames** ({fps:.1f} FPS, **{duration_sec:.1f} seconds**).")

        col_act1, col_act2 = st.columns([1, 2])
        with col_act1:
            run_scan = st.button("▶️ Play & Scan Video Expressions Now")

        if run_scan:
            st.markdown("---")
            st.markdown("##### 🛰️ Real-Time Playback & Expression Scanner")

            col_vid_play, col_live_metrics = st.columns([1.2, 1.0])

            with col_vid_play:
                frame_placeholder = st.empty()
                live_status_bar = st.empty()

            with col_live_metrics:
                live_diag_box = st.empty()
                live_just_sci = st.empty()
                live_just_lay = st.empty()

            timeline_records = []
            annotated_frames = []
            frame_idx = 0
            # Sample every 3 frames for smooth playback + high accuracy
            step = max(1, int(fps / 10))

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            t_start = time.time()
            if hasattr(pipeline, "reset_tracker"):
                pipeline.reset_tracker()
            elif hasattr(pipeline, "face_tracker"):
                pipeline.face_tracker.reset()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                t_sec = frame_idx / fps

                if frame_idx % step == 0:
                    multi_res = pipeline.process_frame_multi(frame, timestamp=t_sec)
                    if multi_res and len(multi_res) > 0:
                        annotated = pipeline.draw_hud_multi(frame, multi_res, show_mesh=show_mesh, show_gaze_rays=show_gaze_rays)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(annotated_rgb, use_container_width=True)
                        annotated_frames.append((t_sec, annotated, multi_res))

                        # Multi-Person Live Status Bar
                        chips_html = "".join([
                            f"<span style='background:#0f172a; border:1px solid #334155; padding:3px 8px; border-radius:4px; margin-left:6px; font-size:0.80rem; color:{'#86efac' if r.affect.sincerity_score>=0.70 else ('#fca5a5' if r.affect.is_masking_detected else '#93c5fd')};'>P{r.track_id}: {int(r.affect.sincerity_score*100)}% ({r.affect.primary_state[:12]})</span>"
                            for r in multi_res
                        ])
                        live_status_bar.markdown(f"""
                        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; font-family: monospace; font-size: 0.85rem; color: #94a3b8; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                            <span>⏱️ TIME: {t_sec:.1f}s / {duration_sec:.1f}s | 👥 {len(multi_res)} FACE{'S' if len(multi_res)>1 else ''} TRACKED</span>
                            <div>{chips_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Multi-Person Live Diagnosis Cards
                        diag_cards_html = ""
                        for r in multi_res:
                            badge_cls = "badge-love" if "Love" in r.affect.primary_state else ("badge-mask" if r.affect.is_masking_detected else ("badge-genuine" if r.affect.sincerity_score >= 0.70 else "badge-neutral"))
                            diag_cards_html += f"""
                            <div class="diag-box" style="padding: 12px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: 600; color: #f8fafc;">👤 Person {r.track_id}</span>
                                    <div class="badge-simple {badge_cls}" style="margin: 0;">{r.affect.primary_state}</div>
                                </div>
                                <div style="font-size: 0.84rem; margin-top: 4px;"><strong>Apparent:</strong> {r.affect.surface_expression}</div>
                                <div style="font-size: 0.84rem; margin-top: 2px;"><strong>Underlying:</strong> {r.affect.underlying_truth}</div>
                                <div style="font-size: 0.80rem; color: #94a3b8; margin-top: 4px;">Sincerity: {int(r.affect.sincerity_score*100)}% | Gaze: {r.gaze.avg_pitch:+.1f}° ({r.gaze.gaze_direction})</div>
                            </div>
                            """
                        live_diag_box.markdown(diag_cards_html, unsafe_allow_html=True)

                        # Dual justifications for Person 1
                        p1 = multi_res[0]
                        live_just_sci.markdown(f"""
                        <div class="just-box" style="padding: 10px 14px; margin-bottom: 8px;">
                            <div class="just-title" style="font-size: 0.75rem;">Scientific Telemetry (P{p1.track_id})</div>
                            <div class="just-body" style="font-size: 0.82rem;">{p1.affect.scientific_justification}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        live_just_lay.markdown(f"""
                        <div class="just-box" style="padding: 10px 14px; margin-bottom: 8px;">
                            <div class="just-title" style="font-size: 0.75rem;">What This Means (P{p1.track_id})</div>
                            <div class="just-body" style="font-size: 0.82rem;">{p1.affect.layman_justification}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        for r in multi_res:
                            timeline_records.append({
                                "person_id": f"Person {r.track_id}",
                                "track_id": r.track_id,
                                "time_sec": round(t_sec, 2),
                                "sincerity_score": r.affect.sincerity_score,
                                "primary_state": r.affect.primary_state,
                                "surface_expression": r.affect.surface_expression,
                                "underlying_truth": r.affect.underlying_truth,
                                "is_masking": r.affect.is_masking_detected,
                                "smile_au12": r.aus.au12_lip_corner_puller,
                                "cheek_au6": r.aus.au6_cheek_raiser,
                                "gaze_pitch": r.gaze.avg_pitch,
                                "gaze_direction": r.gaze.gaze_direction,
                                "duchenne_coherence": r.aus.duchenne_coherence,
                                "blink_bpm": r.blink.blinks_per_minute,
                                "pulse_bpm": r.pulse.bpm,
                                "scientific": r.affect.scientific_justification,
                                "layman": r.affect.layman_justification
                            })

                frame_idx += 1

            cap.release()
            os.unlink(tfile.name)

            if timeline_records:
                st.session_state["video_timeline"] = timeline_records
                st.session_state["annotated_frames"] = annotated_frames
                st.success("✅ Full multi-face video analysis complete!")

        # Display Completed Video Analysis Results if available
        if "video_timeline" in st.session_state and st.session_state["video_timeline"]:
            records = st.session_state["video_timeline"]
            df_time = pd.DataFrame(records)

            unique_persons = sorted(
                df_time["person_id"].unique(),
                key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 1
            )
            num_people = len(unique_persons)

            st.markdown("---")
            st.markdown(f"### 📊 Comprehensive Forensic Report of the Video ({num_people} Individual{'s' if num_people > 1 else ''} Tracked)")

            # 1. Executive Summary Cards
            if num_people == 1:
                s1, s2, s3, s4 = st.columns(4)
                avg_sincerity = df_time["sincerity_score"].mean() * 100
                mask_count = df_time["is_masking"].sum()
                dominant_state = df_time["primary_state"].mode()[0]
                top_smile = df_time["smile_au12"].max() * 100

                with s1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Dominant Feeling</div>
                        <div class="metric-value" style="font-size: 1.15rem;">{dominant_state}</div>
                        <div class="metric-sub">Most Frequent State</div>
                    </div>
                    """, unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Average Sincerity</div>
                        <div class="metric-value" style="color: {'#86efac' if avg_sincerity >= 70 else '#fca5a5'};">{avg_sincerity:.0f}%</div>
                        <div class="metric-sub">Across All Frames</div>
                    </div>
                    """, unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Masking Episodes</div>
                        <div class="metric-value" style="color: {'#fca5a5' if mask_count > 0 else '#86efac'};">{mask_count}</div>
                        <div class="metric-sub">Incongruence Dips</div>
                    </div>
                    """, unsafe_allow_html=True)
                with s4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Peak Smile</div>
                        <div class="metric-value">{top_smile:.0f}%</div>
                        <div class="metric-sub">Max Zygomaticus AU12</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                p_cols = st.columns(num_people)
                for idx, p in enumerate(unique_persons):
                    p_df = df_time[df_time["person_id"] == p]
                    p_avg_sinc = p_df["sincerity_score"].mean() * 100
                    p_mask = p_df["is_masking"].sum()
                    p_dom = p_df["primary_state"].mode()[0] if not p_df.empty else "N/A"
                    with p_cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label" style="font-size:0.85rem; color:#38bdf8; font-weight:600;">👤 {p}</div>
                            <div class="metric-value" style="font-size: 1.15rem; margin-top:4px;">{p_dom}</div>
                            <div class="metric-sub" style="margin-top:6px;">
                                Avg Sincerity: <strong style="color: {'#86efac' if p_avg_sinc >= 70 else '#fca5a5'};">{p_avg_sinc:.0f}%</strong><br>
                                Masking Dips: <strong style="color: {'#fca5a5' if p_mask > 0 else '#86efac'};">{p_mask}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # 2. Interactive View Mode Selector
            if num_people > 1:
                st.markdown("#### 🔍 Select View Mode:")
                view_mode = st.radio(
                    "View Mode",
                    ["👥 Compare All People (Synchronized Timeline)"] + [f"👤 {p}" for p in unique_persons],
                    horizontal=True,
                    label_visibility="collapsed"
                )
            else:
                view_mode = f"👤 {unique_persons[0]}"

            # 3. Synopsis Breakdown
            st.markdown("#### 💡 Forensic Video Breakdown: What Happened in This Clip?")
            if "Compare All" in view_mode or num_people > 1:
                narrative_items = []
                for p in unique_persons:
                    p_df = df_time[df_time["person_id"] == p]
                    p_dom = p_df["primary_state"].mode()[0] if not p_df.empty else "N/A"
                    p_sinc = p_df["sincerity_score"].mean() * 100
                    p_mask = p_df["is_masking"].sum()
                    narrative_items.append(
                        f"<li><strong>{p}:</strong> Displayed predominantly <em>{p_dom}</em> with average sincerity of <strong>{p_sinc:.0f}%</strong> ({p_mask} masking episodes).</li>"
                    )
                st.markdown(f"""
                <div class="just-box">
                    <div class="just-title">Multi-Person Clip Synopsis</div>
                    <div class="just-body">
                        The video contains <strong>{num_people} individuals</strong> tracked simultaneously across the timeline:
                        <ul style="margin-top:6px;">
                            {''.join(narrative_items)}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                p_df = df_time[df_time["person_id"] == unique_persons[0]]
                p_dom = p_df["primary_state"].mode()[0] if not p_df.empty else "N/A"
                p_sinc = p_df["sincerity_score"].mean() * 100
                st.markdown(f"""
                <div class="just-box">
                    <div class="just-title">Clip Synopsis</div>
                    <div class="just-body">
                        Subject displayed predominantly <strong>{p_dom}</strong> with an overall average sincerity index of <strong>{p_sinc:.0f}%</strong>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 4. Interactive Moment-by-Moment Frame Scrubber
            if "annotated_frames" in st.session_state and st.session_state["annotated_frames"]:
                st.markdown("#### 🔍 Interactive Scrubber: Inspect Any Moment")
                ann_list = st.session_state["annotated_frames"]
                max_t = ann_list[-1][0]

                scrub_sec = st.slider("Drag slider to inspect any second in the video:", min_value=0.0, max_value=float(max_t), value=0.0, step=0.1)

                # Find closest frame
                closest_idx = min(range(len(ann_list)), key=lambda i: abs(ann_list[i][0] - scrub_sec))
                chosen_item = ann_list[closest_idx]
                chosen_t = chosen_item[0]
                chosen_img = chosen_item[1]
                chosen_multi = chosen_item[2] if len(chosen_item) > 2 else []

                col_scrub_img, col_scrub_info = st.columns([1.2, 1.0])
                with col_scrub_img:
                    st.image(cv2.cvtColor(chosen_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption(f"Inspecting Frame at {chosen_t:.2f} seconds ({len(chosen_multi)} person(s) present)")

                with col_scrub_info:
                    if chosen_multi:
                        for cr in chosen_multi:
                            st.markdown(f"""
                            <div class="diag-box" style="padding: 12px; margin-bottom: 8px;">
                                <h4 style="margin: 0 0 6px 0; color: #f8fafc;">👤 Person {cr.track_id}: {cr.affect.primary_state}</h4>
                                <div><strong>Sincerity:</strong> {int(cr.affect.sincerity_score * 100)}%</div>
                                <div><strong>Gaze:</strong> {cr.gaze.avg_pitch:+.1f}° ({cr.gaze.gaze_direction})</div>
                                <div><strong>Smile Pull (AU12):</strong> {cr.aus.au12_lip_corner_puller:.2f} | <strong>Cheek (AU6):</strong> {cr.aus.au6_cheek_raiser:.2f}</div>
                                <div style="font-size: 0.80rem; color: #94a3b8; margin-top: 4px;">{cr.affect.layman_justification[:140]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No face tracked at this moment.")

            # 5. Sincerity & Expression Kinetics Timeline Chart
            st.markdown("#### 📈 Sincerity & Expression Kinetics Timeline")
            fig_time = go.Figure()

            palette = ["#38bdf8", "#f472b6", "#a78bfa", "#34d399", "#fbbf24", "#f87171"]

            if "Compare All" in view_mode or (num_people > 1 and "Person" not in view_mode):
                # Multi-line comparative timeline
                for idx, p in enumerate(unique_persons):
                    p_df = df_time[df_time["person_id"] == p].sort_values("time_sec")
                    p_color = palette[idx % len(palette)]
                    fig_time.add_trace(go.Scatter(
                        x=p_df["time_sec"],
                        y=p_df["sincerity_score"] * 100,
                        mode="lines+markers",
                        name=f"{p} Sincerity (%)",
                        line=dict(color=p_color, width=2.5),
                        hovertemplate=f"<b>{p}</b><br>Time: %{{x:.2f}}s<br>Sincerity: %{{y:.0f}}%<extra></extra>"
                    ))

                    # Masking points
                    p_mask_df = p_df[p_df["is_masking"] == True]
                    if not p_mask_df.empty:
                        fig_time.add_trace(go.Scatter(
                            x=p_mask_df["time_sec"],
                            y=p_mask_df["sincerity_score"] * 100,
                            mode="markers",
                            name=f"{p} Masked State",
                            marker=dict(color=p_color, size=9, symbol="x"),
                            hovertemplate=f"<b>{p} Masking Dip</b><br>Time: %{{x:.2f}}s<extra></extra>"
                        ))
            else:
                # Individual person deep-dive curve
                chosen_p = view_mode.replace("👤 ", "").strip()
                p_df = df_time[df_time["person_id"] == chosen_p].sort_values("time_sec")

                fig_time.add_trace(go.Scatter(
                    x=p_df["time_sec"],
                    y=p_df["sincerity_score"] * 100,
                    mode="lines+markers",
                    name="Sincerity (%)",
                    line=dict(color="#38bdf8", width=2.5)
                ))
                fig_time.add_trace(go.Scatter(
                    x=p_df["time_sec"],
                    y=p_df["smile_au12"] * 100,
                    mode="lines",
                    name="Smile (AU12 %)",
                    line=dict(color="#f59e0b", width=1.5, dash="dot")
                ))
                fig_time.add_trace(go.Scatter(
                    x=p_df["time_sec"],
                    y=p_df["gaze_pitch"],
                    mode="lines",
                    name="Gaze Pitch (Deg)",
                    line=dict(color="#a78bfa", width=1.5)
                ))
                p_mask_df = p_df[p_df["is_masking"] == True]
                if not p_mask_df.empty:
                    fig_time.add_trace(go.Scatter(
                        x=p_mask_df["time_sec"],
                        y=p_mask_df["sincerity_score"] * 100,
                        mode="markers",
                        name="Masked State Detected",
                        marker=dict(color="#ef4444", size=9, symbol="x")
                    ))

            fig_time.update_layout(
                xaxis_title="Time (seconds)",
                yaxis_title="Sincerity (%) / Degrees",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                hovermode="x unified",
                height=380,
                margin=dict(l=20, r=20, t=30, b=30)
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # 6. Dyadic / Interpersonal Emotional Dynamics (when multiple faces present)
            if num_people >= 2:
                st.markdown("#### 🤝 Interpersonal Emotional Synchrony")
                p1_name, p2_name = unique_persons[0], unique_persons[1]
                p1_df = df_time[df_time["person_id"] == p1_name].set_index("time_sec")
                p2_df = df_time[df_time["person_id"] == p2_name].set_index("time_sec")
                shared_times = p1_df.index.intersection(p2_df.index)

                if len(shared_times) > 2:
                    p1_sinc = p1_df.loc[shared_times, "sincerity_score"]
                    p2_sinc = p2_df.loc[shared_times, "sincerity_score"]
                    correlation = np.corrcoef(p1_sinc, p2_sinc)[0, 1]
                    corr_str = f"{correlation:+.2f}" if not np.isnan(correlation) else "N/A"

                    st.markdown(f"""
                    <div class="just-box">
                        <div class="just-title">Dyadic Synchrony Metric</div>
                        <div class="just-body">
                            Sincerity Trajectory Correlation (<strong>{p1_name}</strong> vs <strong>{p2_name}</strong>): <strong>{corr_str}</strong><br>
                            {"✨ <strong>Strong Emotional Attunement:</strong> Both individuals show coordinated, harmonious sincerity trajectories." if (not np.isnan(correlation) and correlation > 0.4) else ("⚡ <strong>Affective Divergence:</strong> The subjects exhibit distinct or asynchronous emotional trajectories." if (not np.isnan(correlation) and correlation < -0.2) else "⚖️ <strong>Independent Affect:</strong> Both individuals maintain distinct internal emotional baselines.")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # 7. Timeline Data Table & JSON Export
            st.markdown("#### 📋 Timestamp Records")
            filtered_df = df_time if ("Compare All" in view_mode or num_people == 1) else df_time[df_time["person_id"] == view_mode.replace("👤 ", "").strip()]
            display_cols = [c for c in ["person_id", "time_sec", "primary_state", "sincerity_score", "is_masking", "gaze_pitch", "smile_au12"] if c in filtered_df.columns]
            st.dataframe(filtered_df[display_cols], use_container_width=True)

            all_records_json = records
            st.download_button(
                label="Download Full Timeline Forensics JSON",
                data=json.dumps(all_records_json, indent=2),
                file_name="affectsense_video_timeline_forensics.json",
                mime="application/json",
                key="dl_timeline_json"
            )



# =========================================================================
# TAB 4: 25+ EMOTION CATALOG & SCIENCE
# =========================================================================
with tab_science:
    st.markdown("##### 25+ Human Affective States")
    st.write("Summary of detectable human affective states and their biometric cues:")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        **1. Love, Romance & Warm Attachment**
        - **In Love / Deep Affection**: Sustained eye lock, gentle AU12 smile, relaxed brow, elevated pulse.
        - **Shy Attraction / Flirting**: Intermittent coy glance away/down, micro-smile, playful blink rate.
        - **Compassionate Tenderness**: Soft eye aperture, subtle AU1 compassion lift, gentle smile.
        - **Playful Delight**: Open mouth smile, lively micro-saccades, head tilt.

        **2. Masked & Forced States (Core)**
        - **Masked Sadness ("Smiling Melancholy")**: Smile on lips + downward retinal gaze + grief brow (AU1).
        - **Forced / Pan Am Smile**: Smile with unengaged orbital cheeks (AU6 near zero).
        - **Masked Anxiety**: Smile paired with rapid blink flutter bursts (>30 BPM).
        - **Contempt / Smugness**: Unilateral asymmetric lip pull (AU14 / asymmetric AU12).
        - **Suppressed Frustration**: Outward smile masking furrowed corrugator brows (AU4).
        - **Jealousy / Envious Resentment**: Hardened gaze lock, corrugator furrow, tight suppressed micro-smile.
        """)

    with c2:
        st.markdown("""
        **3. Vulnerable & Inward Senses**
        - **Genuine Sadness / Grief**: Grief brow (AU1) + corrugator (AU4) + lip depression (AU15) + downward gaze.
        - **Shame / Dejection / Guilt**: Persistent downward and sideways retinal deflection.
        - **Embarrassment / Flustered**: Downward gaze with nervous micro-smile, elevated pulse.
        - **Nostalgia / Bittersweet Melancholy**: Mild smile with distant/elevated gaze.
        - **Anxiety & Restlessness**: Erratic micro-saccades, rapid blinks, scanning pupils.
        - **Suppressed Anger**: Corrugator furrow + pressed lips (AU24) + rigid stare.
        - **Fear / Panic**: Wide eye aperture (AU5) + raised brows + elevated pulse.
        - **Astonishment / Surprise**: Wide eyes + eyebrow elevation + dropped jaw.
        - **Boredom / Apathy**: Heavy drooping eyelids, sluggish saccades, flat affect.

        **4. Cognitive & Basal States**
        - **Deep Concentration**: Micro-fixated gaze, narrowed eye aperture.
        - **Serene Contentment**: Soft bilateral mouth relaxation, calm steady gaze.
        - **Pride / Triumph**: Elevated chin pitch, broad symmetrical smile.
        - **Confusion / Skepticism**: Asymmetric brow tension.
        - **Neutral Baseline**: Balanced symmetry, relaxed facial muscles.
        """)

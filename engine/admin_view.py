"""
AffectSense AI - Streamlit Admin Key Tracking & Analytics Dashboard
===================================================================
Provides a secure administrative interface for repository owners to:
  - Monitor all issued Access Keys and active sessions in MongoDB
  - Inspect claimed user identities (Name, Phone, Email)
  - Revoke compromised or unauthorized keys in real-time
  - Export audit logs to CSV
  - View instructions for local-only key generation CLI
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Any

def render_admin_dashboard(auth_mgr: Any):
    """Renders the comprehensive, dark-mode Admin Dashboard."""
    
    # 1. Header Banner & Diagnostics
    metrics = auth_mgr.get_admin_dashboard_metrics()
    is_mongo = metrics.get("mongo_connected", False)
    db_name = metrics.get("mongo_db_name", "affectsense_ai")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border: 1px solid #334155; border-radius: 12px; padding: 22px 26px; margin-bottom: 24px; box-shadow: 0 4px 20px -2px rgba(0,0,0,0.4);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.4); color: #38bdf8; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; padding: 3px 10px; border-radius: 9999px; letter-spacing: 0.05em;">
                    Security Command Center
                </span>
                <h1 style="margin: 8px 0 4px 0; font-size: 1.8rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
                    🔑 Access Key & Identity Tracker
                </h1>
                <p style="margin: 0; font-size: 0.88rem; color: #94a3b8;">
                    Centralized verification records, active session lifecycles, and user credentials tracked in MongoDB.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Database Connectivity Status Strip & Refresh
    col_status, col_btn = st.columns([3, 1])
    with col_status:
        if is_mongo:
            st.markdown(f"""
            <div style="display: inline-flex; align-items: center; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; padding: 6px 14px; margin-bottom: 18px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; margin-right: 8px; box-shadow: 0 0 8px #10b981;"></span>
                <span style="font-size: 0.82rem; color: #a7f3d0; font-weight: 600;">MongoDB Cloud Active</span>
                <span style="font-size: 0.76rem; color: #6ee7b7; margin-left: 8px; font-family: monospace;">(cluster0.hi21i1o.mongodb.net / {db_name})</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: inline-flex; align-items: center; background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; padding: 6px 14px; margin-bottom: 18px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #f59e0b; margin-right: 8px;"></span>
                <span style="font-size: 0.82rem; color: #fde68a; font-weight: 600;">Local Cache Mode</span>
                <span style="font-size: 0.76rem; color: #fcd34d; margin-left: 8px;">(MongoDB connection pending or offline)</span>
            </div>
            """, unsafe_allow_html=True)
    with col_btn:
        if st.button("🔄 Refresh Data", key="btn_refresh_admin", use_container_width=True):
            st.rerun()

    # 3. KPI Metrics Grid
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px 12px; text-align: center;">
            <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Total Keys</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; font-family: monospace; margin-top: 4px;">{metrics.get('total_keys', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #166534; border-radius: 10px; padding: 14px 12px; text-align: center;">
            <div style="font-size: 0.7rem; color: #86efac; text-transform: uppercase; font-weight: 600;">Active Sessions</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #4ade80; font-family: monospace; margin-top: 4px;">{metrics.get('active_sessions', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #854d0e; border-radius: 10px; padding: 14px 12px; text-align: center;">
            <div style="font-size: 0.7rem; color: #fde047; text-transform: uppercase; font-weight: 600;">Unused Pending</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #facc15; font-family: monospace; margin-top: 4px;">{metrics.get('unused_keys', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px 12px; text-align: center;">
            <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Expired Keys</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #cbd5e1; font-family: monospace; margin-top: 4px;">{metrics.get('expired_keys', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #991b1b; border-radius: 10px; padding: 14px 12px; text-align: center;">
            <div style="font-size: 0.7rem; color: #fca5a5; text-transform: uppercase; font-weight: 600;">Revoked Keys</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #f87171; font-family: monospace; margin-top: 4px;">{metrics.get('revoked_keys', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #0369a1; border-radius: 10px; padding: 14px 12px; text-align: center;">
            <div style="font-size: 0.7rem; color: #7dd3fc; text-transform: uppercase; font-weight: 600;">Users Registered</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8; font-family: monospace; margin-top: 4px;">{metrics.get('total_registered_users', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # 4. Filter & Search Controls
    col_search, col_f_role, col_f_status = st.columns([2.2, 1.2, 1.2])
    with col_search:
        search_query = st.text_input(
            "Search Database",
            placeholder="🔍 Search by key, user name, phone, email, or label...",
            key="admin_search_box"
        )
    with col_f_role:
        role_choice = st.selectbox(
            "Filter Role",
            ["All Roles", "Admins Only", "Users Only"],
            key="admin_role_select"
        )
    with col_f_status:
        status_choice = st.selectbox(
            "Filter Status",
            ["All Statuses", "Active", "Unused", "Expired", "Revoked"],
            key="admin_status_select"
        )

    # Convert select choices to API parameters
    role_param = "all"
    if role_choice == "Admins Only":
        role_param = "admin"
    elif role_choice == "Users Only":
        role_param = "user"

    status_param = "all"
    if status_choice != "All Statuses":
        status_param = status_choice.lower()

    # Query filtered keys
    keys = auth_mgr.list_keys(
        role_filter=role_param,
        status_filter=status_param,
        search_term=search_query
    )

    st.markdown(f"**Showing {len(keys)} records**")

    # 5. Tabbed View: Live Table vs Detailed Key Cards
    tab_table, tab_cards, tab_generator = st.tabs(["📋 Key Directory Table", "🔍 Detailed Key Records & Actions", "🛠️ Local Generator CLI Guide"])

    # TAB 1: DataFrame Table
    with tab_table:
        if not keys:
            st.info("No matching keys found in database. Try adjusting your search or filters.")
        else:
            table_rows = []
            for k in keys:
                table_rows.append({
                    "Key": k.get("key") or k.get("display_prefix"),
                    "Role": k.get("role", "user").upper(),
                    "Status": k.get("status", "").upper(),
                    "User Name": k.get("user_name") or "-",
                    "Phone": k.get("user_phone") or "-",
                    "Email": k.get("user_email") or "-",
                    "Duration (min)": k.get("duration_minutes", 10),
                    "Created At": (k.get("created_at") or "")[:19].replace("T", " "),
                    "Used At": (k.get("used_at") or "-")[:19].replace("T", " "),
                    "Expires At": (k.get("expires_at") or "-")[:19].replace("T", " "),
                    "Label": k.get("label", ""),
                })
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Export to CSV
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Filtered Records (CSV)",
                data=csv_bytes,
                file_name=f"affectsense_keys_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="btn_download_csv"
            )

    # TAB 2: Detailed Cards with Instant Revocation
    with tab_cards:
        if not keys:
            st.info("No matching records found.")
        else:
            st.caption("Click on any record to inspect complete verification details, timestamps, or to revoke access.")
            for idx, k in enumerate(keys):
                st_badge = k.get("status", "unused").upper()
                r_badge = k.get("role", "user").upper()
                u_name = k.get("user_name") or "Unclaimed"
                key_token = k.get("key") or k.get("display_prefix")
                
                status_color = {
                    "ACTIVE": "#4ade80",
                    "UNUSED": "#facc15",
                    "EXPIRED": "#94a3b8",
                    "REVOKED": "#f87171"
                }.get(st_badge, "#38bdf8")

                expander_title = f"[{st_badge}] [{r_badge}] {key_token}  •  {u_name}"
                with st.expander(expander_title, expanded=(idx < 2)):
                    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
                    with c1:
                        st.markdown("**🔑 Key Details**")
                        st.code(key_token, language="text")
                        st.markdown(f"**Role:** `{r_badge}`")
                        st.markdown(f"**Label:** `{k.get('label') or '-'}`")
                        st.markdown(f"**Duration:** `{k.get('duration_minutes', 10)} mins`")
                    with c2:
                        st.markdown("**👤 User Identity**")
                        st.markdown(f"**Full Name:** `{k.get('user_name') or 'Not entered'}`")
                        st.markdown(f"**Phone:** `{k.get('user_phone') or 'Not entered'}`")
                        st.markdown(f"**Email:** `{k.get('user_email') or 'Not entered'}`")
                        st.markdown(f"**Redemption Count:** `{k.get('redemption_count', 0)}`")
                    with c3:
                        st.markdown("**⏱️ Timestamps**")
                        st.markdown(f"**Created:** `{(k.get('created_at') or '-')[:19].replace('T', ' ')}`")
                        st.markdown(f"**Used At:** `{(k.get('used_at') or '-')[:19].replace('T', ' ')}`")
                        st.markdown(f"**Expires At:** `{(k.get('expires_at') or '-')[:19].replace('T', ' ')}`")
                        
                        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
                        if k.get("status") != "revoked":
                            if st.button(f"🚨 Revoke Key", key=f"btn_revoke_{k.get('key_hash') or idx}", use_container_width=True):
                                target = k.get("key") or k.get("key_hash")
                                revoked = auth_mgr.revoke_key(target)
                                if revoked:
                                    st.success(f"Key {k.get('display_prefix')} has been revoked.")
                                    st.rerun()
                                else:
                                    st.error("Failed to revoke key.")
                        else:
                            st.markdown("<span style='color: #f87171; font-weight: 700;'>⛔ REVOKED</span>", unsafe_allow_html=True)

    # TAB 3: Local Generator CLI Instructions
    with tab_generator:
        st.markdown("""
        ### 🛡️ Local-Only Key Generation Architecture
        
        To prevent pirating, unauthorized cloning, and leakage to open source, **key generation is intentionally restricted to your local machine** and is **NEVER** exposed in public Streamlit builds.
        
        `make_key.py` is excluded from git via `.gitignore` so your generation credentials and logic remain private.
        
        #### How to Issue Keys Locally:
        Run the following commands in your local project terminal:
        """)
        
        st.code("""
# 1. Mint 1 single-use 10-minute User Access Key (default):
python make_key.py

# 2. Mint a User Key with custom duration (e.g. 30 minutes):
python make_key.py --minutes 30 --label "Client Demo"

# 3. Mint multiple User Keys at once (e.g. batch of 5):
python make_key.py --count 5 --minutes 15

# 4. Mint a privileged Admin Key (gives full access + admin dashboard):
python make_key.py --admin --label "Co-Founder Alice"

# 5. List all tracked keys and user registrations in your terminal:
python make_key.py --list

# 6. Revoke a compromised key directly from the terminal:
python make_key.py --revoke AFTS-XXXX-XXXX-XXXX-XXXX
        """, language="bash")
        
        st.info("💡 Keys created with `python make_key.py` are immediately synced to MongoDB and appear in real time on this dashboard.")

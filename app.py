import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# =========================
# Constants
# =========================
rulebook_left = -0.83083
rulebook_right = 0.83083
rulebook_bottom = 1.5
rulebook_top = 3.3775

shadow_left = -0.99750
shadow_right = 0.99750
shadow_bottom = 1.377
shadow_top = 3.5

strike_zone_middle_x = (rulebook_left + rulebook_right) / 2
strike_zone_middle_y = (rulebook_bottom + rulebook_top) / 2

x_splits = np.linspace(rulebook_left, rulebook_right, 4)
y_splits = np.linspace(rulebook_bottom, rulebook_top, 4)

pitch_marker_map = {
    "Fastball": "circle",
    "Sinker": "circle",
    "Cutter": "triangle-up",
    "Slider": "triangle-up",
    "Curveball": "triangle-up",
    "Sweeper": "triangle-up",
    "Splitter": "square",
    "ChangeUp": "square"
}

# =========================
# Data
# =========================
sec_csv_path = "SEC_Pitching_pbp_cleaned_for_catchers.csv"
fawley_csv_path = "Fall_2025_wRV_with_stuff.csv"

columns_needed = ['Batter', 'BatterSide', 'Pitcher', 'PitcherThrows',
                  'Catcher', 'PitchCall', 'TaggedPitchType',
                  'PlateLocSide', 'PlateLocHeight', 'Date']

rebs_columns_needed = ['Batter', 'BatterSide', 'Pitcher', 'PitcherThrows',
                       'Catcher', 'PitchCall', 'TaggedPitchType',
                       'PlateLocSide', 'PlateLocHeight', 'Date',
                       'Inning', 'Balls', 'Strikes', 'PitcherTeam',
                       # If present, we’ll use this for CSAA:
                       'ProbStrikeCalled']

df_sec = pd.read_csv(sec_csv_path, usecols=columns_needed)
df_fawley = pd.read_csv(fawley_csv_path, usecols=[c for c in rebs_columns_needed if c in pd.read_csv(fawley_csv_path, nrows=0).columns])

# Keep only called pitches for framing context
df_sec = df_sec[df_sec['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]
df_fawley = df_fawley[df_fawley['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]

# Dates → datetime
df_fawley['Date'] = pd.to_datetime(df_fawley['Date'], errors='coerce')
df_sec['Date'] = pd.to_datetime(df_sec['Date'], errors='coerce')
df_fawley = df_fawley.dropna(subset=['Date'])
df_sec = df_sec.dropna(subset=['Date'])

# =========================
# UI
# =========================
st.title("2025 Ole Miss Catcher Reports")

report_tab, leaderboard_tab = st.tabs(["Report", "Leaderboard"])

with report_tab:
    # Sidebar-like controls (inside Report tab)
    catcher_options = df_fawley['Catcher'].dropna().unique()
    selected_catcher = st.selectbox("Select a Catcher:", catcher_options)

    date_options = pd.to_datetime(df_fawley['Date']).dropna().unique()
    date_range = st.date_input(
        "Select Date Range:",
        [date_options.min(), date_options.max()]
    )

    batter_side_options = ["All"] + df_fawley['BatterSide'].dropna().unique().tolist()
    selected_batter_side = st.selectbox("Select Batter Side:", batter_side_options)

    pitcher_throws_options = ["All"] + df_fawley['PitcherThrows'].dropna().unique().tolist()
    selected_pitcher_throws = st.selectbox("Select Pitcher Throws:", pitcher_throws_options)

    pitch_categories = {
        "All Pitches" : ["Curveball", "Cutter", "Slider", "Sweeper", "Fastball", "Sinker", "ChangeUp", "Splitter"],
        "Fast/Sink": ["Fastball", "Sinker"],
        "Breaking Ball": ["Curveball", "Cutter", "Slider", "Sweeper"],
        "Change/Split": ["ChangeUp", "Splitter"]
    }
    selected_pitch_category = st.selectbox("Select a Pitch Type Category:", options=pitch_categories.keys())

    # Filter data based on selections
    filtered_fawley = df_fawley[df_fawley['Catcher'] == selected_catcher].copy()
    filtered_fawley = filtered_fawley[
        (pd.to_datetime(filtered_fawley['Date']) >= pd.Timestamp(date_range[0])) &
        (pd.to_datetime(filtered_fawley['Date']) <= pd.Timestamp(date_range[1]))
    ]
    if selected_batter_side != "All":
        filtered_fawley = filtered_fawley[filtered_fawley['BatterSide'] == selected_batter_side]
    if selected_pitcher_throws != "All":
        filtered_fawley = filtered_fawley[filtered_fawley['PitcherThrows'] == selected_pitcher_throws]
    if selected_pitch_category:
        valid_pitch_types = pitch_categories[selected_pitch_category]
        filtered_fawley = filtered_fawley[filtered_fawley['TaggedPitchType'].isin(valid_pitch_types)]

    # Derived subsets
    strike_pitches_df = filtered_fawley[filtered_fawley["PitchCall"] == "StrikeCalled"]
    ball_pitches_df   = filtered_fawley[filtered_fawley["PitchCall"] == "BallCalled"]
    all_pitches_df    = filtered_fawley.copy()
    shadow_pitches_df = filtered_fawley[
        ((filtered_fawley["PlateLocSide"] < rulebook_left) | (filtered_fawley["PlateLocSide"] > rulebook_right)) |
        ((filtered_fawley["PlateLocHeight"] < rulebook_bottom) | (filtered_fawley["PlateLocHeight"] > rulebook_top))
    ]

    def calculate_strike_percentage(df_):
        return 0.0 if len(df_) == 0 else (len(df_[df_["PitchCall"] == "StrikeCalled"]) / len(df_)) * 100

    strike_percentage_strike = 100.0
    strike_percentage_ball = 0.0
    strike_percentage_all = calculate_strike_percentage(all_pitches_df)
    strike_percentage_shadow = calculate_strike_percentage(shadow_pitches_df)

    def get_marker_shape(pitch_type):
        return pitch_marker_map.get(pitch_type, "diamond")

    def create_zone_scatter(title, pitch_df):
        fig = go.Figure()
        for _, row in pitch_df.iterrows():
            color = "green" if row["PitchCall"] == "StrikeCalled" else "red"
            marker_shape = get_marker_shape(row["TaggedPitchType"])
            fig.add_trace(go.Scatter(
                x=[row["PlateLocSide"]],
                y=[row["PlateLocHeight"]],
                mode="markers",
                marker=dict(symbol=marker_shape, color=color, size=8, line=dict(color='black', width=1.5)),
                showlegend=False,
                hoverinfo="text",
                text=f"Inning: {row.get('Inning','')}" +
                     f"<br>Balls: {row.get('Balls','')}" +
                     f"<br>Strikes: {row.get('Strikes','')}" +
                     f"<br>Pitcher: {row.get('Pitcher','')}" +
                     f"<br>Pitch Type: {row.get('TaggedPitchType','')}" +
                     f"<br>Batter: {row.get('Batter','')}" +
                     f"<br>BatterSide: {row.get('BatterSide','')}"+
                     f"<br>ProbStrikeCalled: {round(row.get('ProbStrikeCalled', 0), 3):.3f}"

            ))

        # Strike zone grid
        for i in range(4):
            fig.add_shape(type="line", x0=x_splits[i], x1=x_splits[i], y0=rulebook_bottom, y1=rulebook_top, line=dict(color="black", width=1))
            fig.add_shape(type="line", x0=rulebook_left, x1=rulebook_right, y0=y_splits[i], y1=y_splits[i], line=dict(color="black", width=1))

        # Shadow zone
        fig.add_shape(type="rect", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_top,
                      line=dict(color="blue", width=2, dash="dash"))

        # Connectors for shadow box
        fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=shadow_bottom, y1=rulebook_bottom,
                      line=dict(color="blue", width=2, dash="dash"))
        fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=rulebook_top, y1=shadow_top,
                      line=dict(color="blue", width=2, dash="dash"))
        fig.add_shape(type="line", x0=shadow_left, x1=rulebook_left, y0=strike_zone_middle_y, y1=strike_zone_middle_y,
                      line=dict(color="blue", width=2, dash="dash"))
        fig.add_shape(type="line", x0=rulebook_right, x1=shadow_right, y0=strike_zone_middle_y, y1=strike_zone_middle_y,
                      line=dict(color="blue", width=2, dash="dash"))
        fig.add_shape(type="line", x0=shadow_left, x1=shadow_right, y0=shadow_top, y1=shadow_top,
                      line=dict(color="blue", width=2, dash="dash"))
        fig.add_shape(type="line", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_bottom,
                      line=dict(color="blue", width=2, dash="dash"))

        fig.update_layout(
            title=title,
            xaxis=dict(range=[-2.5, 2.5], title="PlateLocSide"),
            yaxis=dict(range=[0.5, 4.5], title="PlateLocHeight"),
            showlegend=False,
            width=400, height=400
        )
        return fig

    def plot_called_strike_kde(df_, title):
        strikes = df_[df_["PitchCall"] == "StrikeCalled"]
        total = df_.copy()

        strike_coords = strikes[["PlateLocSide", "PlateLocHeight"]].dropna()
        total_coords = total[["PlateLocSide", "PlateLocHeight"]].dropna()

        if len(total_coords) < 20 or len(strike_coords) < 5:
            st.warning("Not enough pitch data to generate a called strike rate map.")
            return go.Figure()

        xy_strike = np.vstack([strike_coords["PlateLocSide"], strike_coords["PlateLocHeight"]])
        xy_total = np.vstack([total_coords["PlateLocSide"], total_coords["PlateLocHeight"]])

        if np.std(xy_total[0]) < 1e-4 or np.std(xy_total[1]) < 1e-4:
            st.warning("Pitch location data is too uniform to create a contour map.")
            return go.Figure()

        kde_strike = gaussian_kde(xy_strike, bw_method=0.2)
        kde_total = gaussian_kde(xy_total, bw_method=0.2)

        x_grid = np.linspace(-2.0, 2.0, 100)
        y_grid = np.linspace(0.5, 4.5, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        coords = np.vstack([X.ravel(), Y.ravel()])

        Z_strike = kde_strike(coords).reshape(X.shape)
        Z_total = kde_total(coords).reshape(X.shape)

        with np.errstate(divide='ignore', invalid='ignore'):
            CSR = np.divide(Z_strike, Z_total)
            CSR[Z_total < 1e-6] = np.nan

        fig = go.Figure(data=go.Heatmap(
            z=CSR, x=x_grid, y=y_grid, colorscale='Jet',
            zmin=0, zmax=1, colorbar=dict(title='Called Strike Rate')
        ))
        fig.add_shape(type="rect", x0=rulebook_left, x1=rulebook_right, y0=rulebook_bottom, y1=rulebook_top,
                      line=dict(color="black", width=2))
        fig.add_shape(type="rect", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_top,
                      line=dict(color="black", width=2, dash="dot"))
        fig.update_layout(
            title=title,
            xaxis=dict(title="PlateLocSide", range=[-2.0, 2.0]),
            yaxis=dict(title="PlateLocHeight", range=[0.5, 4.5]),
            width=500, height=500
        )
        return fig

    # =========================
    # Report content
    # =========================
    total_pitches = len(all_pitches_df)
    st.markdown(f"**Total Pitches Caught:** {total_pitches}")

    fig1 = create_zone_scatter(f"StrikeCalled Pitches (Strike%: {strike_percentage_strike:.1f}%)", strike_pitches_df)
    fig2 = create_zone_scatter(f"BallCalled Pitches (Strike%: {strike_percentage_ball:.1f}%)", ball_pitches_df)
    fig3 = create_zone_scatter(f"All Pitches (Strike%: {strike_percentage_all:.1f}%)", all_pitches_df)
    fig4 = create_zone_scatter(f"Shadow Zone Pitches (Strike%: {strike_percentage_shadow:.1f}%)", shadow_pitches_df)

    st.write(f"### {selected_catcher} Framing Breakdown:")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    col3.plotly_chart(fig3, use_container_width=True)
    col4.plotly_chart(fig4, use_container_width=True)

    # ===== Framing Performance table (with CSAA) =====
    def calculate_framing_metrics(df_):
        rows = []

        # Balls Called Strikes (outside rulebook but called strike)
        bcs = df_[(((df_['PlateLocSide'] < rulebook_left) | (df_['PlateLocSide'] > rulebook_right)) |
                   ((df_['PlateLocHeight'] < rulebook_bottom) | (df_['PlateLocHeight'] > rulebook_top))) &
                  (df_['PitchCall'] == 'StrikeCalled')].shape[0]

        # Strikes Called Balls (inside rulebook but called ball)
        scb = df_[((df_['PlateLocSide'] >= rulebook_left) & (df_['PlateLocSide'] <= rulebook_right) &
                   (df_['PlateLocHeight'] >= rulebook_bottom) & (df_['PlateLocHeight'] <= rulebook_top) &
                   (df_['PitchCall'] == 'BallCalled'))].shape[0]

        # 50/50 zone
        fifty = df_[(((df_['PlateLocSide'] >= shadow_left) & (df_['PlateLocSide'] <= shadow_right)) &
                      ((df_['PlateLocHeight'] >= shadow_bottom) & (df_['PlateLocHeight'] <= shadow_top))) &
                    ~(((df_['PlateLocSide'] >= rulebook_left) & (df_['PlateLocSide'] <= rulebook_right)) &
                      ((df_['PlateLocHeight'] >= rulebook_bottom) & (df_['PlateLocHeight'] <= rulebook_top)))]
        fifty_total = len(fifty)
        fifty_strikes = fifty[fifty['PitchCall'] == 'StrikeCalled'].shape[0]
        fifty_display = f"{fifty_strikes} / {fifty_total}"

        rows.append(["Balls Called Strikes", bcs])
        rows.append(["Strikes Called Balls", scb])
        rows.append(["50/50 Pitches", fifty_display])

        # ---- CSAA & CSAA/100 (called pitches only) ----
        if "ProbStrikeCalled" in df_.columns:
            called = df_[df_["PitchCall"].isin(["BallCalled", "StrikeCalled"])].copy()
            called = called[called["ProbStrikeCalled"].notna()]
            if len(called) > 0:
                called["is_strike"] = (called["PitchCall"] == "StrikeCalled").astype(int)
                csaa = float((called["is_strike"] - called["ProbStrikeCalled"]).sum())
                csaa_per_100 = 100.0 * csaa / max(len(called), 1)
                rows.append(["CSAA (sum)", f"{csaa:.2f}"])
                rows.append(["CSAA per 100", f"{csaa_per_100:.2f}"])
            else:
                rows.append(["CSAA (sum)", "N/A"])
                rows.append(["CSAA per 100", "N/A"])
        else:
            st.info("`ProbStrikeCalled` not found in the data — CSAA metrics unavailable.")
            rows.append(["CSAA (sum)", "N/A"])
            rows.append(["CSAA per 100", "N/A"])

        return rows

    framing_table = calculate_framing_metrics(filtered_fawley)
    st.write("### Framing Performance")
    st.table(framing_table)

    st.write("### Called Strike Rate Contour Map")
    csr_kde_fig = plot_called_strike_kde(all_pitches_df, f"CSR Contour: {selected_catcher}")
    st.plotly_chart(csr_kde_fig, use_container_width=True)

with leaderboard_tab:
    st.header("Leaderboards")
    st.info("Coming soon…")

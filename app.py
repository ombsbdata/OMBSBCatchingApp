import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Define constants for the strike zone
rulebook_left = -0.83083
rulebook_right = 0.83083
rulebook_bottom = 1.5
rulebook_top = 3.3775

# Define the new shadow zone boundaries
shadow_left = -0.99750
shadow_right = 0.99750
shadow_bottom = 1.377
shadow_top = 3.5

# Define middle of the strike zone
strike_zone_middle_x = (rulebook_left + rulebook_right) / 2
strike_zone_middle_y = (rulebook_bottom + rulebook_top) / 2

# Define 9 even zones inside the original strike zone
x_splits = np.linspace(rulebook_left, rulebook_right, 4)
y_splits = np.linspace(rulebook_bottom, rulebook_top, 4)

# Load CSVs
sec_csv_path = "SEC_Pitching_pbp_cleaned_for_catchers.csv"
fawley_csv_path = "Brewster_2025_MASTER.CSV"
df = "Brewster_2025_MASTER.CSV"

# Load datasets
columns_needed = ['Batter', 'BatterSide', 'Pitcher', 'PitcherThrows',
                  'Catcher', 'PitchCall', 'TaggedPitchType',
                  'PlateLocSide', 'PlateLocHeight', 'Date']

rebs_columns_needed = ['Batter', 'BatterSide', 'Pitcher', 'PitcherThrows',
                  'Catcher', 'PitchCall', 'TaggedPitchType',
                  'PlateLocSide', 'PlateLocHeight', 'Date','Inning','Balls','Strikes','PitcherTeam']
df_sec = pd.read_csv(sec_csv_path, usecols=columns_needed)
df_fawley = pd.read_csv(fawley_csv_path, usecols=rebs_columns_needed)
df = pd.read_csv(fawley_csv_path, usecols = rebs_columns_needed)

# Filter for relevant PitchCalls
df_sec = df_sec[df_sec['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]
df_fawley = df_fawley[df_fawley['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]
#df_fawley = df_fawley[df_fawley['PitcherTeam'] == 'OLE_REB']

# Streamlit UI
st.title("2025 Ole Miss Catcher Reports")

# Convert Date column to datetime format to handle different formats
df_fawley['Date'] = pd.to_datetime(df_fawley['Date'], errors='coerce')
df_sec['Date'] = pd.to_datetime(df_sec['Date'], errors='coerce')

# Drop rows where Date conversion failed (if necessary)
df_fawley = df_fawley.dropna(subset=['Date'])
df_sec = df_sec.dropna(subset=['Date'])


# Catcher selection
catcher_options = df_fawley['Catcher'].dropna().unique()
selected_catcher = st.selectbox("Select a Catcher:", catcher_options)

# Date selection
date_options = pd.to_datetime(df_fawley['Date']).dropna().unique()
date_range = st.date_input("Select Date Range:", [date_options.min(), date_options.max()])

# Batter Side filter (Dropdown: All, Right, Left)
batter_side_options = ["All"] + df_fawley['BatterSide'].dropna().unique().tolist()
selected_batter_side = st.selectbox("Select Batter Side:", batter_side_options)

# Pitcher Throws filter (Dropdown: All, Right, Left)
pitcher_throws_options = ["All"] + df_fawley['PitcherThrows'].dropna().unique().tolist()
selected_pitcher_throws = st.selectbox("Select Pitcher Throws:", pitcher_throws_options)


pitch_categories = {
    "All Pitches" : ["Curveball", "Cutter", "Slider", "Sweeper","Fastball", "Sinker","ChangeUp", "Splitter"],
    "Fast/Sink": ["Fastball", "Sinker"],
    "Breaking Ball": ["Curveball", "Cutter", "Slider", "Sweeper"],
    "Change/Split": ["ChangeUp", "Splitter"]
}



# Pitch Type Category filter
selected_pitch_category = st.selectbox("Select a Pitch Type Category:", options=pitch_categories.keys())

# Filter data based on user selections
filtered_fawley = df_fawley[df_fawley['Catcher'] == selected_catcher]

# Apply date filtering
filtered_fawley = filtered_fawley[
    (pd.to_datetime(filtered_fawley['Date']) >= pd.Timestamp(date_range[0])) &
    (pd.to_datetime(filtered_fawley['Date']) <= pd.Timestamp(date_range[1]))
]

# Apply BatterSide filter
if selected_batter_side != "All":
    filtered_fawley = filtered_fawley[filtered_fawley['BatterSide'] == selected_batter_side]

# Apply PitcherThrows filter
if selected_pitcher_throws != "All":
    filtered_fawley = filtered_fawley[filtered_fawley['PitcherThrows'] == selected_pitcher_throws]

# Apply Pitch Type Category filter
if selected_pitch_category:
    valid_pitch_types = pitch_categories[selected_pitch_category]
    filtered_fawley = filtered_fawley[filtered_fawley['TaggedPitchType'].isin(valid_pitch_types)]

# Prepare datasets for each plot using the updated filtered data
strike_pitches_df = filtered_fawley[filtered_fawley["PitchCall"] == "StrikeCalled"]
ball_pitches_df = filtered_fawley[filtered_fawley["PitchCall"] == "BallCalled"]
all_pitches_df = filtered_fawley.copy()
shadow_pitches_df = filtered_fawley[
    ((filtered_fawley["PlateLocSide"] < rulebook_left) | (filtered_fawley["PlateLocSide"] > rulebook_right)) |
    ((filtered_fawley["PlateLocHeight"] < rulebook_bottom) | (filtered_fawley["PlateLocHeight"] > rulebook_top))
]

def calculate_strike_percentage(df):
    if len(df) == 0:
        return 0.0  # Avoid division by zero
    return (len(df[df["PitchCall"] == "StrikeCalled"]) / len(df)) * 100


# Compute Strike% BEFORE using them in titles
strike_percentage_strike = 100.0  # Since this plot only contains StrikeCalled pitches
strike_percentage_ball = 0.0  # Since this plot only contains BallCalled pitches
strike_percentage_all = calculate_strike_percentage(all_pitches_df)
strike_percentage_shadow = calculate_strike_percentage(shadow_pitches_df)


from scipy.stats import gaussian_kde

from scipy.stats import gaussian_kde

def plot_called_strike_kde(df, title):
    # Filter down to strikes and all pitches
    strikes = df[df["PitchCall"] == "StrikeCalled"]
    total = df.copy()

    # Extract coordinates and drop NaNs
    strike_coords = strikes[["PlateLocSide", "PlateLocHeight"]].dropna()
    total_coords = total[["PlateLocSide", "PlateLocHeight"]].dropna()

    # Validate pitch volume
    if len(total_coords) < 20 or len(strike_coords) < 5:
        st.warning("Not enough pitch data to generate a called strike rate map.")
        return go.Figure()

    # Convert to arrays
    xy_strike = np.vstack([strike_coords["PlateLocSide"], strike_coords["PlateLocHeight"]])
    xy_total = np.vstack([total_coords["PlateLocSide"], total_coords["PlateLocHeight"]])

    # Validate spatial variance
    if np.std(xy_total[0]) < 1e-4 or np.std(xy_total[1]) < 1e-4:
        st.warning("Pitch location data is too uniform to create a contour map.")
        return go.Figure()

    # KDE smoothing
    kde_strike = gaussian_kde(xy_strike, bw_method=0.2)
    kde_total = gaussian_kde(xy_total, bw_method=0.2)

    # Grid setup
    x_grid = np.linspace(-2.0, 2.0, 100)
    y_grid = np.linspace(0.5, 4.5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    coords = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate KDEs on grid
    Z_strike = kde_strike(coords).reshape(X.shape)
    Z_total = kde_total(coords).reshape(X.shape)

    # Compute Called Strike Rate
    with np.errstate(divide='ignore', invalid='ignore'):
        CSR = np.divide(Z_strike, Z_total)
        CSR[Z_total < 1e-6] = np.nan  # Mask empty areas

    # Create heatmap plot
    fig = go.Figure(data=go.Heatmap(
        z=CSR,
        x=x_grid,
        y=y_grid,
        colorscale='Jet',
        zmin=0, zmax=1,
        colorbar=dict(title='Called Strike Rate')
    ))

    # Overlay strike zone and shadow zone
    fig.add_shape(type="rect", x0=rulebook_left, x1=rulebook_right, y0=rulebook_bottom, y1=rulebook_top,
                  line=dict(color="black", width=2))
    fig.add_shape(type="rect", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_top,
                  line=dict(color="black", width=2, dash="dot"))

    # Layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="PlateLocSide", range=[-2.0, 2.0]),
        yaxis=dict(title="PlateLocHeight", range=[0.5, 4.5]),
        width=500, height=500
    )

    return fig



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


# Function to get marker shape based on pitch type
def get_marker_shape(pitch_type):
    return pitch_marker_map.get(pitch_type, "diamond")  # Default to rhombus (diamond) for "Other"

def create_zone_scatter(title, pitch_df):
    fig = go.Figure()

    # Add scatter plot for pitches with hover tooltips
    for index, row in pitch_df.iterrows():
        color = "green" if row["PitchCall"] == "StrikeCalled" else "red"
        marker_shape = get_marker_shape(row["TaggedPitchType"])

        fig.add_trace(go.Scatter(
            x=[row["PlateLocSide"]],
            y=[row["PlateLocHeight"]],
            mode="markers",
            marker=dict(symbol=marker_shape, color=color, size=8, line=dict(color='black', width=1.5)),
            showlegend=False,
            hoverinfo="text",
            text=f"Inning: {row['Inning']}<br>"
                 f"Balls: {row['Balls']}<br>"
                 f"Strikes: {row['Strikes']}<br>"
                 f"Pitcher: {row['Pitcher']}<br>"
                 f"Pitch Type: {row['TaggedPitchType']}<br>"
                 f"Batter: {row['Batter']}<br>"
                 f"BatterSide: {row['BatterSide']}"
        ))

    # Draw main strike zone
    for i in range(4):
        fig.add_shape(type="line", x0=x_splits[i], x1=x_splits[i], y0=rulebook_bottom, y1=rulebook_top, line=dict(color="black", width=1))
        fig.add_shape(type="line", x0=rulebook_left, x1=rulebook_right, y0=y_splits[i], y1=y_splits[i], line=dict(color="black", width=1))

    # Draw shadow zone outlines
    fig.add_shape(type="rect", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_top,
                  line=dict(color="blue", width=2, dash="dash"))

    # Ensure horizontal and vertical dashed lines align with strike zone
    fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=shadow_bottom, y1=rulebook_bottom,
                  line=dict(color="blue", width=2, dash="dash"))
    fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=rulebook_top, y1=shadow_top,
                  line=dict(color="blue", width=2, dash="dash"))

    fig.add_shape(type="line", x0=shadow_left, x1=rulebook_left, y0=strike_zone_middle_y, y1=strike_zone_middle_y,
                  line=dict(color="blue", width=2, dash="dash"))
    fig.add_shape(type="line", x0=rulebook_right, x1=shadow_right, y0=strike_zone_middle_y, y1=strike_zone_middle_y,
                  line=dict(color="blue", width=2, dash="dash"))

    # Ensure full enclosure with horizontal lines
    fig.add_shape(type="line", x0=shadow_left, x1=shadow_right, y0=shadow_top, y1=shadow_top,
                  line=dict(color="blue", width=2, dash="dash"))
    fig.add_shape(type="line", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_bottom,
                  line=dict(color="blue", width=2, dash="dash"))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-2.5, 2.5], title="PlateLocSide"),
        yaxis=dict(range=[0.5, 4.5], title="PlateLocHeight"),
        showlegend=False,
        width=400, height=400
    )

    return fig

# Display total pitch count caught by the selected catcher under current filters
total_pitches = len(all_pitches_df)
st.markdown(f"**Total Pitches Caught:** {total_pitches}")


# Create individual plots with updated filtering
fig1 = create_zone_scatter(f"StrikeCalled Pitches (Strike%: {strike_percentage_strike:.1f}%)", strike_pitches_df)
fig2 = create_zone_scatter(f"BallCalled Pitches (Strike%: {strike_percentage_ball:.1f}%)", ball_pitches_df)
fig3 = create_zone_scatter(f"All Pitches (Strike%: {strike_percentage_all:.1f}%)", all_pitches_df)
fig4 = create_zone_scatter(f"Shadow Zone Pitches (Strike%: {strike_percentage_shadow:.1f}%)", shadow_pitches_df)

# Streamlit layout
st.write(f"### {selected_catcher} Framing Breakdown:")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)
col3.plotly_chart(fig3, use_container_width=True)
col4.plotly_chart(fig4, use_container_width=True)


# Streamlit layout







def calculate_framing_metrics(df):
    """Calculates framing performance metrics for the selected dataset."""
    
    # Balls Called Strikes: Pitches OUTSIDE rulebook zone that were called strikes
    balls_called_strikes = df[
        (((df['PlateLocSide'] < rulebook_left) | (df['PlateLocSide'] > rulebook_right)) |  
         ((df['PlateLocHeight'] < rulebook_bottom) | (df['PlateLocHeight'] > rulebook_top))) 
        & (df['PitchCall'] == 'StrikeCalled')
    ].shape[0]

    # Strikes Called Balls: Pitches INSIDE rulebook zone that were called balls
    strikes_called_balls = df[
        ((df['PlateLocSide'] >= rulebook_left) & (df['PlateLocSide'] <= rulebook_right)) &
        ((df['PlateLocHeight'] >= rulebook_bottom) & (df['PlateLocHeight'] <= rulebook_top)) &
        (df['PitchCall'] == 'BallCalled')
    ].shape[0]

    # 50/50 Pitches: Pitches that are BETWEEN the rulebook zone and the shadow zone
    fifty_fifty_pitches = df[
        (((df['PlateLocSide'] >= shadow_left) & (df['PlateLocSide'] <= shadow_right)) &  
         ((df['PlateLocHeight'] >= shadow_bottom) & (df['PlateLocHeight'] <= shadow_top)))  # Inside shadow zone
        & 
        ~(((df['PlateLocSide'] >= rulebook_left) & (df['PlateLocSide'] <= rulebook_right)) &  
          ((df['PlateLocHeight'] >= rulebook_bottom) & (df['PlateLocHeight'] <= rulebook_top)))  # Outside rulebook zone
    ]

    # Total pitches in 50/50 zone
    total_fifty_fifty_pitches = fifty_fifty_pitches.shape[0]
    
    # Pitches in 50/50 zone that were called strikes
    total_fifty_fifty_strikes = fifty_fifty_pitches[fifty_fifty_pitches['PitchCall'] == 'StrikeCalled'].shape[0]

    # Format as "x / y"
    fifty_fifty_display = f"{total_fifty_fifty_strikes} / {total_fifty_fifty_pitches}"

    # Return table data
    return [
        ["Balls Called Strikes", balls_called_strikes],
        ["Strikes Called Balls", strikes_called_balls],
        ["50/50 Pitches", fifty_fifty_display]
    ]

# Compute the framing table dynamically based on user-selected filters
framing_table = calculate_framing_metrics(filtered_fawley)

# Display in Streamlit
st.write("### Framing Performance")
st.table(framing_table)
st.write("### Called Strike Rate Contour Map")
csr_kde_fig = plot_called_strike_kde(all_pitches_df, f"CSR Contour: {selected_catcher}")
st.plotly_chart(csr_kde_fig, use_container_width=True)




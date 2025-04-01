import humanize
import warnings
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px
from streamlit_js_eval import get_geolocation
from constants import LANDMARK_COLORS, MAPBOX_STYLES
from utils import hex_to_rgb

# Load the original DataFrame
og_df = pd.read_csv("assets/data/all_routes_combined.csv")
df = og_df.copy()
df['size'] = 5

def map_config(data, landmark_color):
    layer = pdk.Layer(
        'ScatterplotLayer',
        data,
        get_position='[longitude, latitude]',
        get_radius=50,
        get_color=hex_to_rgb(landmark_color),
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=data['latitude'].mean(),
        longitude=data['longitude'].mean(),
        zoom=10,
        pitch=10
    )

    tooltip = {
        "html": "<b>Station:</b> {station}",
        "style": {"color": "white"}
    }
    
    return layer, view_state, tooltip

def main(map):
    global df
    
    a1, a2, a3, a4, input_area, locate_me = map.columns([0.65, 0.65, 0.65, 1, 1.6, 0.5])

    try:
        cols = map.columns([0.3, 1])

        with cols[0]:
            st.write('<h4 class="poppins-light">Dashboard</h4>', unsafe_allow_html=True)
            locate_me = st.checkbox("Locate Me", key="locate_me")
            mapbox_style = st.selectbox("Map Style", MAPBOX_STYLES, key="mapbox_style")
                
            user_lat, user_long = None, None  # Initialize user location variables
            user_location_df = pd.DataFrame(columns=['latitude', 'longitude', 'color', 'size'])  # Initialize empty user location DataFrame

            if locate_me:
                loc = get_geolocation()
                if loc is not None:
                    st.toast("📍 Locating you...")
                    user_lat, user_long = loc['coords']['latitude'], loc['coords']['longitude']
                    st.toast("📍 Location found! {} {}".format(user_lat, user_long))
                    
                    # Create a new DataFrame for user location
                    user_location_df = pd.DataFrame({
                        'latitude': [user_lat],
                        'longitude': [user_long],
                        'color': ['#00ff00'],  # Green for the user location
                        'size': [20]
                    })

                    # Display fetched location
                    st.write(f"**Latitude:** {user_lat}")
                    st.write(f"**Longitude:** {user_long}")

        with cols[1]:
            # Display the main map with the original DataFrame
            layer, view_state, tooltip = map_config(df, landmark_color="#ff0000")
            map = pdk.Deck(
                map_style=MAPBOX_STYLES[mapbox_style],
                initial_view_state=view_state,
                layers=[layer],
                tooltip=tooltip,
            )
            st.pydeck_chart(map)

            # If user location exists, create a separate map for it
            if not user_location_df.empty:
                user_layer, user_view_state, user_tooltip = map_config(user_location_df, landmark_color="#00ff00")  # Green for user location
                user_map = pdk.Deck(
                    map_style=MAPBOX_STYLES[mapbox_style],
                    initial_view_state=pdk.ViewState(
                        latitude=user_lat,
                        longitude=user_long,
                        zoom=12,  # Closer zoom for user location
                        pitch=10
                    ),
                    layers=[user_layer],
                    tooltip=user_tooltip,
                )
                st.subheader("Your Location")
                st.pydeck_chart(user_map)  # Display the user location map

    except Exception as e:
        st.error(e)

if __name__ == '__main__':
    main(st)

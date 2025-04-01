import humanize
import warnings
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px
from streamlit_js_eval import get_geolocation
from constants import LANDMARK_COLORS

def generate_map(df, selected_route):
    arc_data = []
    scatter_data = []
    df = df[df['route'] == selected_route] if selected_route != 'All' else df

    for i in range(len(df) - 1):
        arc_data.append({
            "from_name": df.iloc[i]["station"],
            "to_name": df.iloc[i + 1]["station"],
            "from_coordinates": [df.iloc[i]["longitude"], df.iloc[i]["latitude"]],
            "to_coordinates": [df.iloc[i + 1]["longitude"], df.iloc[i + 1]["latitude"]],
            "route": df.iloc[i]["route"]
        })
        
        scatter_data.append({
            "coordinates": [df.iloc[i]["longitude"], df.iloc[i]["latitude"]],
            "station_name": df.iloc[i]["station"]
        })

    scatter_data.append({
        "coordinates": [df.iloc[-1]["longitude"], df.iloc[-1]["latitude"]],
        "station_name": df.iloc[-1]["station"]
    })

    arc_layer = pdk.Layer(
        "ArcLayer",
        arc_data,
        pickable=True,
        get_source_position="from_coordinates",
        get_target_position="to_coordinates",
        get_source_color=LANDMARK_COLORS[selected_route]['source'],
        get_target_color=LANDMARK_COLORS[selected_route]['target'],
        get_width=3,
        auto_highlight=True,
        tooltip={"text": "{from_name} to {to_name} - Route: {route}"}
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        scatter_data,
        pickable=True,
        get_position="coordinates",
        get_fill_color=LANDMARK_COLORS[selected_route]['rgb'],
        get_radius=100,
        tooltip={"text": "{station_name}"},
    )

    view_state = pdk.ViewState(
        latitude=df['latitude'].mean(), 
        longitude=df['longitude'].mean(), 
        zoom=11, 
        bearing=0, 
        pitch=50
    )

    r = pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view_state,
    )

    return r


def main(overview):
    overview.write("""<h3><i class="fa-solid fa-chart-simple"></i>&nbsp;&nbsp;Overview Of BRTS Dataset</h3>""", unsafe_allow_html=True)
    a1, a2, a3, a4, input_area, dl_csv = overview.columns([0.65,0.65,0.65,0.8,1.35,0.6])

    try:
        og_df = pd.read_csv("assets/data/all_routes_combined.csv", index_col=0)
        
        df = og_df.copy()
        df['size'] = 5
        df['color'] = df['route'].map(lambda x: LANDMARK_COLORS[x]['rgb'])
        
        selected_route = input_area.selectbox("Select a Route", [*LANDMARK_COLORS, 'All'])

        a1.metric("Total Stations", df.shape[0])
        a2.metric(":blue[SR Routes]", 8, help="Standard Routes")
        a3.metric(":blue[TR Routes]", 4, help="Trunk Routes")
        a4.metric(":green[Selected Route Stations]", df.shape[0] if selected_route == 'All' else df[df['route'] == selected_route].shape[0])

        csv = og_df.to_csv(index=False).encode('utf-8')
        dl_csv.write("<br>", unsafe_allow_html=True)
        dl_csv.download_button("Download Data", data=csv, file_name="all_routes_combined.csv", mime="text/csv", use_container_width=True)
        
        cols = overview.columns([1,0.7])

        with cols[0]:
            st.dataframe(og_df if selected_route == 'All' else og_df.loc[df['route']==selected_route], height=420, use_container_width=True)
        with cols[1]:
            with st.container():
                map = generate_map(df, selected_route)
                st.components.v1.html(map.to_html(as_string=True), height=420) 
                # st.pydeck_chart(map)
                # st.map(df if selected_route == 'All' else df.loc[df['route']==selected_route],
                #     color='color',
                #     size='size',
                #     height=390
                # )
            
    except IndexError:
        overview.dataframe(df, height=200)

if __name__ == '__main__':
    main()
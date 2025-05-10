import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv("data/data.csv")

    data = data.drop(["Unnamed: 32", "id"], axis=1)

    data['diagnosis'] = data['diagnosis'].map({ 'M' : 1, 'B' : 0 })

    return data



def add_sidebar():
    st.sidebar.header("Cell Nuclei Details")

    data = get_clean_data()

    # label and the corresponding column name
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # we need to extract the key values that are in the sidebar to visualize it onto the chart so we use a dictionary
    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label, 
            min_value=float(data[key].min()), 
            max_value=float(data[key].max()), 
            value=float(data[key].mean())
        ) 

    return input_dict


def get_scaled_values(input_data):
  data = get_clean_data()

  X = data.drop(["diagnosis"], axis=1)

  scaled_dict = {}

  for key, value in input_data.items():
    max_value = X[key].max()
    min_value = X[key].min()
    scaled_value = (value - min_value) / (max_value - min_value)
    scaled_dict[key] = scaled_value

  return scaled_dict


def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig

def add_prediction(input_data):
  model = joblib.load("model/model.pkl")
  scaler = joblib.load("model/scaler.pkl")

  # converting the values of the key into single array of values
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  # what it does it takes the values of the cell neclei measurement value we have choosen and the place those value
  # into an array (here we have 30 values so 30 values will be placed into an array)
  # and then reshape it to 1 row and 30 columns into vertical form

  input_array_scaled = scaler.transform(input_array)

  # here we are predicting if its benign or malignant tumor with being 0 or 1
  prediction = model.predict(input_array_scaled)

  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is classified as:")

  if prediction[0] == 0:
    st.success("Benign")
  else:
     st.error("Malicious")

  st.write("Probability of being benign", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being malicious", model.predict_proba(input_array_scaled)[0][1])


def main():
  st.set_page_config(
    page_title="Breast Cancer Detection", 
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )

  input_data = add_sidebar()

  with st.container():
    st.title("Breast Cancer Detection")
    st.write("This is a simple web app to predict whether a tumor is benign or malignant based on its features.")

    col1, col2 = st.columns([4,1]) # the first colum is 4 times bigger than the second column
    
  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)

  with col2:
    add_prediction(input_data)


if __name__ == "__main__":
    main()


# python app/main.py
# streamlit run app/main.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn




# Load the machine learning model
with open('breast_cancer_model.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Function to make predictions
def make_prediction(features, scale_factor=1):
    features_array = np.array(features).reshape(1, -1)
    # Scale the features for prediction
    features_array[:, :4] *= scale_factor
    prediction = loaded_model.predict(features_array)
    print("Raw Prediction:", prediction)
    return prediction[0]

# Function to simulate processing
def simulate_processing():
    progress_bar = st.progress(0)
    for i in range(100):
        # Update the progress bar
        progress_bar.progress(i + 1)

# Streamlit App

# Add custom CSS for a pink theme
pink_css = """
    <style>
        body {
            background-color: #FFC0CB; /* Light Pink */
        }
        .sidebar .sidebar-content {
            background-color: #FFC0CB; /* Light Pink */
        }
    </style>
"""

st.markdown(pink_css, unsafe_allow_html=True)



image_url = 'https://th.bing.com/th/id/OIP.nqlmOzQymCdXw3Mhx8BCqQHaE8?w=270&h=180&c=7&r=0&o=5&dpr=1.5&pid=1.7'
st.image('logo.jpg', use_column_width=True)

st.title("HOW DOES CANCER GUARDIAN WORKS")


# Data for the steps
steps_data = [
    {"Step": "Data Collection and Preprocessing", "Description": "We have compiled a diverse and comprehensive dataset of Fine Needle Aspiration (FNA) reports, including samples from both benign and malignant cases. The dataset undergoes preprocessing to standardize and enhance the quality of the data, ensuring optimal performance of the machine learning model."},
    {"Step": "Machine Learning Model", "Description": "Our project employs a sophisticated machine learning model trained on the preprocessed FNA report data. The model has learned to recognize patterns and features indicative of cancerous cells, making predictions based on the input FNA report."},
    {"Step": "Feature Extraction", "Description": "The FNA report contains vital information about cell morphology, structure, and other characteristics. The machine learning model extracts relevant features from the FNA report, transforming the raw data into meaningful representations."},
    {"Step": "Prediction and Classification", "Description": "The model utilizes the extracted features to make predictions regarding the nature of the cells in the FNA report. The primary classification is between benign and malignant cells, providing crucial information for early cancer detection."},
    {"Step": "Interpretation and User Interface", "Description": "The project includes an intuitive and user-friendly interface for inputting FNA reports. Users receive clear and concise results, indicating whether the FNA report suggests a likelihood of malignancy."},
]

# Display the steps in a table

st.table(steps_data)


# Input form for user to enter parameters
st.sidebar.header("Input Parameters")
input_parameters = {}

# Provided values
provided_values = [8.196, 16.84, 51.71, 201.9, 0.086, 0.05943, 0.01588, 0.005917, 0.1769, 0.06503,
                   0.1563, 0.9567, 1.094, 8.205, 0.008968, 0.01646, 0.01588, 0.005917, 0.02574,
                   0.002582, 8.964, 21.96, 57.26, 242.2, 0.1297, 0.1357, 0.0688, 0.02564, 0.3105, 0.07409]



# Assuming X is a DataFrame containing the column names you provided
# This is just a placeholder for the column names
X_columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
             'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
             'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
             'area error', 'smoothness error', 'compactness error', 'concavity error',
             'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius',
             'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
             'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
             'worst fractal dimension']


# Customized ranges for each slider based on provided values
slider_ranges = {
    'mean radius': (0, 50) if provided_values[0] == int(provided_values[0]) else (0.0, 50.0),
    'mean texture': (0, 50) if provided_values[1] == int(provided_values[1]) else (0.0, 50.0),
    'mean perimeter': (0, 200) if provided_values[2] == int(provided_values[2]) else (0.0, 200.0),
    'mean area': (0, 2000) if provided_values[3] == int(provided_values[3]) else (0.0, 2000.0),
    'mean smoothness': (0, 0.5) if provided_values[4] == int(provided_values[4]) else (0.0, 0.5),
    'mean compactness': (0, 0.5) if provided_values[5] == int(provided_values[5]) else (0.0, 0.5),
    'mean concavity': (0, 0.5) if provided_values[6] == int(provided_values[6]) else (0.0, 0.5),
    'mean concave points': (0, 0.5) if provided_values[7] == int(provided_values[7]) else (0.0, 0.5),
    'mean symmetry': (0, 0.5) if provided_values[8] == int(provided_values[8]) else (0.0, 0.5),
    'mean fractal dimension': (0, 0.1) if provided_values[9] == int(provided_values[9]) else (0.0, 0.1),
    'radius error': (0, 2) if provided_values[10] == int(provided_values[10]) else (0.0, 2.0),
    'texture error': (0, 2) if provided_values[11] == int(provided_values[11]) else (0.0, 2.0),
    'perimeter error': (0, 20) if provided_values[12] == int(provided_values[12]) else (0.0, 20.0),
    'area error': (0, 200) if provided_values[13] == int(provided_values[13]) else (0.0, 200.0),
    'smoothness error': (0, 0.01) if provided_values[14] == int(provided_values[14]) else (0.0, 0.01),
    'compactness error': (0, 0.1) if provided_values[15] == int(provided_values[15]) else (0.0, 0.1),
    'concavity error': (0, 0.1) if provided_values[16] == int(provided_values[16]) else (0.0, 0.1),
    'concave points error': (0, 0.1) if provided_values[17] == int(provided_values[17]) else (0.0, 0.1),
    'symmetry error': (0, 0.1) if provided_values[18] == int(provided_values[18]) else (0.0, 0.1),
    'fractal dimension error': (0, 0.01) if provided_values[19] == int(provided_values[19]) else (0.0, 0.01),
    'worst radius': (0, 100) if provided_values[20] == int(provided_values[20]) else (0.0, 100.0),
    'worst texture': (0, 100) if provided_values[21] == int(provided_values[21]) else (0.0, 100.0),
    'worst perimeter': (0, 400) if provided_values[22] == int(provided_values[22]) else (0.0, 400.0),
    'worst area': (0, 4000) if provided_values[23] == int(provided_values[23]) else (0.0, 4000.0),
    'worst smoothness': (0, 1) if provided_values[24] == int(provided_values[24]) else (0.0, 1.0),
    'worst compactness': (0, 1) if provided_values[25] == int(provided_values[25]) else (0.0, 1.0),
    'worst concavity': (0, 1) if provided_values[26] == int(provided_values[26]) else (0.0, 1.0),
    'worst concave points': (0, 1) if provided_values[27] == int(provided_values[27]) else (0.0, 1.0),
    'worst symmetry': (0, 1) if provided_values[28] == int(provided_values[28]) else (0.0, 1.0),
    'worst fractal dimension': (0, 0.5) if provided_values[29] == int(provided_values[29]) else (0.0, 0.5),
}
# Scale factor for prediction (adjust based on the actual data range)
scale_factor = 0.01


for i, column in enumerate(X_columns):
    # Use provided values as default slider values
    input_parameters[column] = st.sidebar.slider(f"{column}:", min_value=slider_ranges[column][0],
                                                 max_value=slider_ranges[column][1],
                                                value=provided_values[i])



# Submit button to trigger prediction
if st.sidebar.button("Submit"):
    # Make prediction using the input parameters
    prediction = make_prediction(list(input_parameters.values()), scale_factor=scale_factor)
    print("Raw Prediction:", prediction)

     # Display the input parameters
    st.header("The Parameters submitted by you")
    input_df = pd.DataFrame([input_parameters])
    st.write(input_df)

    # Display the prediction
    st.subheader("And here is your Prediction:")
    if prediction == 0:
        st.write('The Breast cancer is Malignant')
    else:
        st.write('The Breast Cancer is Benign')

     # Adding a progress bar
    progress_bar = st.progress(0)

    # Simulate processing
    for i in range(100):
        # Update the progress bar
        progress_bar.progress(i + 1)

    # Display a success message
    st.success("Processing complete!")

    
    


# Footer section
st.markdown("<hr class='footer'>", unsafe_allow_html=True)

# About Us
st.markdown("<h3>About Us</h3>", unsafe_allow_html=True)
st.write("Hey there! This breast cancer detection project was made by <br> ZAINAB <br>KHUSHI KHARI<br>SOUMYA SHUBHAM<br>AARUSH SACDEVA",unsafe_allow_html=True)

# More Info about Breast Cancer and This Project
st.markdown("[More Info about Breast Cancer and This Project](www.google.com)", unsafe_allow_html=True)


# End of footer
st.markdown("<hr class='footer'>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2023 Breast Cancer Detection App</p>", unsafe_allow_html=True)


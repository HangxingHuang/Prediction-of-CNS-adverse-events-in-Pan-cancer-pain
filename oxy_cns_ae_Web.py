import streamlit as st
from joblib import load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from scipy import stats

st.title("XGBoost-based Prediction of central nervous system adverse events in Pan-cancer pain patients administered with oxycodone sustained-release formulation")
# Create a function to generate HTML for person icons
def generate_person_icons(filled_count, total_count=100):
    # SVG person icon
    icon_svg = """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="7" r="4" stroke="black" stroke-width="2" fill="none"/>
      <path d="M4 21C4 16.6863 7.68629 13 12 13C16.3137 13 20 16.6863 20 21H4Z" stroke="black" stroke-width="2" fill="none"/>
    </svg>
    """
    
    # Replace fill attribute to change color
    filled_icon = icon_svg.replace('fill="none"', 'fill="lightblue"')
    empty_icon = icon_svg.replace('fill="none"', 'fill="gray"')

    # Generate the HTML for the icons
    icons_html = ''.join([filled_icon if i < filled_count else empty_icon for i in range(total_count)])
    return f"<div style='display: flex; flex-wrap: wrap; width: 480px;'>{icons_html}</div>"
# Load model
loaded_model = load("xgb_pan_cancer_pain_4%_model.joblib")

# Load saved Scaler
scaler = joblib.load('xgb_pan_cancer_pain_4%_scaler.joblib')

# Load validation set predictions
validation_predictions = np.load('xgb_pan_cancer_pain_4%_predictions.npy')
# Ensure validation_predictions is a 1D array
if validation_predictions.ndim > 1:
    validation_predictions = validation_predictions.ravel()

# Define feature order
features = ['WBC', 'TBA', 'BAS%', 'Weight', 'CREA','OPRM1 rs9397685', 'LYM', 'RDW']
continuous_features = ['WBC', 'TBA', 'BAS%', 'Weight', 'CREA', 'LYM', 'RDW']

# Categorical feature mappings
OPRM1_options = {0: 'AA', 1: 'AG', 2: 'GG'}


# Reverse mappings
OPRM1_reverse = {v: k for k, v in OPRM1_options.items()}
# p40_reverse = {v: k for k, v in p40_options.items()}


# Left column: input form
with st.sidebar:
    st.header("Your information")
    
 # Continuous features input
    wbc = st.number_input('White Blood Cell (WBC, 10^9/L)', min_value=0.0, max_value=100.0, step=0.1, key='wbc')
    tba = st.number_input('Total Bile Acid (TBA, µmol/L)', min_value=0.0, max_value=100.0, step=0.1, key='tba')
    bas_percentage = st.number_input('Basophils Percentage (BAS%, %)', min_value=0.0, max_value=100.0, step=0.1, key='bas_percentage')
    weight = st.number_input('Weight (Kg)', min_value=0.0, max_value=500.0, step=1.0, key='weight')
    crea = st.number_input('Creatinine (CREA, µmol/L)', min_value=0.0, max_value=5000.0, step=1.0, key='crea')
    # Categorical feature input
    OPRM1_rs9397685 = st.selectbox('OPRM1 rs9397685 Genotype', options=list(OPRM1_options.values()), key='OPRM1_rs9397685')

    lym = st.number_input('Lymphocytes (LYM, 10^9/L)', min_value=0.0, max_value=50.0, step=0.1, key='lym')
    rdw = st.number_input('Red Cell Distribution Width (RDW, %)', min_value=0.0, max_value=100.0, step=0.1, key='rdw')
    


# Middle column: buttons
with st.container():
    st.write("")  # Placeholder

    # Use custom CSS for button styles
    st.markdown(
        """
        <style>
        .clear-button {
            background-color: transparent;
            color: black;
            border: none;
            text-decoration: underline;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            display: inline-block;
        }
        .clear-button:hover {
            color: red;
        }
        .clear-button:active {
            color: darkred;
        }
        </style>
        """, unsafe_allow_html=True)

    # Use HTML button
    st.markdown(
        """
        <a href="?reset=true" class="clear-button">Clear</a>
        """, unsafe_allow_html=True)

# If the prediction button is clicked
if st.button('Prediction'):
        # Prepare input data
        user_input = pd.DataFrame([[wbc, tba, bas_percentage, weight, crea, OPRM1_reverse[OPRM1_rs9397685],lym,rdw]], columns=features)
        
        # Extract continuous features
        user_continuous_input = user_input[continuous_features]
        
        # Normalize continuous features
        user_continuous_input_normalized = scaler.transform(user_continuous_input)
        
        # Combine normalized data back into the full input
        user_input_normalized = user_input.copy()
        user_input_normalized[continuous_features] = user_continuous_input_normalized

        # Get prediction probability
        prediction_proba = loaded_model.predict_proba(user_input_normalized)[:, 1][0]
        prediction_percentage = round(prediction_proba * 100)

        # Combine user prediction with validation predictions
        combined_predictions = np.concatenate([validation_predictions, np.array([prediction_proba])])

        # Calculate standard deviation and confidence interval
        std_dev = np.std(combined_predictions)
        confidence_level = 0.95
        degrees_of_freedom = len(combined_predictions) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        margin_of_error = t_critical * (std_dev / np.sqrt(len(combined_predictions)))
        lower_bound_percentage = max(prediction_percentage - margin_of_error * 100, 0)
        upper_bound_percentage = min(prediction_percentage + margin_of_error * 100, 100)

        lower_bound_percentage = round(lower_bound_percentage)
        upper_bound_percentage = round(upper_bound_percentage)
        
        # Right column: show prediction results
        with st.container():
            st.header("Your result")
            st.markdown(f"The probability that Pan-cancer pain patients how likely occur central nervous system adverse events is (95% confidence interval):")
            result_html = f"""
            <div style="display: flex; align-items: center;">
                <span style="color:red; font-weight:bold; font-size:48px;">{prediction_percentage}%</span>
                <span style="margin-left: 10px;">({lower_bound_percentage}% to {upper_bound_percentage}%)</span>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)
            # Use the function to generate icons based on prediction
            icons_html = generate_person_icons(prediction_percentage)

            # Display the generated icons
            st.markdown(f"""
                <div style="display: flex; align-items: center;">
                </div>
                <div>
                    {icons_html}
                </div>
            """, unsafe_allow_html=True)
            
            # Show additional information
            st.write(f"This result predicts how likely you are to have a central nervous system adverse event after taking an extended-release formulation of oxycodone. The probability means that out of 100 patients with similar characteristics, approximately {prediction_percentage} % may have a central nervous system adverse event. More specifically, we're 95% confident that {lower_bound_percentage} to {upper_bound_percentage} out of 100 patients may have a central nervous system adverse event, based on our training data. However, it's important to recognize that this is just a rough ballpark estimate. Individual patient outcomes can vary significantly, and a healthcare provider can provide a more precise assessment, taking into account a broader range of factors and personal medical history.")
            st.markdown(f"<span style='color:red;'>Disclaimer:</span> This tool is provided for informational purposes only and should NOT be considered as medical advice or a substitute for professional consultation. Users should seek proper medical counsel and discuss their treatment options with a qualified healthcare provider.", unsafe_allow_html=True)

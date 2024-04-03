import streamlit as st
import pickle
import numpy as np

# load the model and data
df = pickle.load(open('kndf.sav', 'rb'))
xgb_model = pickle.load(open('xgb.sav', 'rb'))

# define options for user selection
BRAND_OPTIONS = ['APPLE', 'ASUS', 'LENOVO', 'HP', 'MSI', 'ACER', 'DELL', 'OTHER', 'AVITA', 'GIGABYTE']
MODEL_OPTIONS = ['MACBOOK', 'VIVOBOOK', 'INSPIRON', 'TUF', 'ASPIRE', 'ZENBOOK', 'LEGION', 'ROG', 'EXPERTBOOK', 'MODERN', 'PAVILION', 'PROBOOK']
CPU_OPTIONS = ['AMD', 'INTEL I7','INTEL I5', 'INTEL I3','OTHERS']
OS_OPTIONS = ['MACOS','WINDOWS', 'LINAX/DOS', 'OTHERS']
DISPLAY_OPTIONS = ['FHD', 'NORMAL', 'LED', 'IPS']
GRAPHICS_OPTIONS = ['INTEL', 'NVIDIA', 'OTHERS', 'AMD']
TOUCH_OPTIONS = ['YES', 'NO']

# title of the model
st.title('LAPTOP PRICE PREDICTOR')

# inputs for user selection
brand = st.selectbox('BRAND', BRAND_OPTIONS)
model = st.selectbox('MODEL', MODEL_OPTIONS)
cpu = st.selectbox('CPU', CPU_OPTIONS)
display = st.selectbox('DISPLAY', DISPLAY_OPTIONS)
touch = st.selectbox('TOUCH', TOUCH_OPTIONS)
ssd = st.selectbox('SSD(GB)', [128, 256, 512, 1024, 2048, 3072])
weight = st.selectbox('WEIGHT', [1.25, 1.50, 1.75, 2, 2.5])
os = st.selectbox('OPERATING SYSTEM', OS_OPTIONS)
adapter = st.selectbox('ADAPTER', df['adapters'].unique())
battery = st.selectbox('BATTERY', df['batteries'].unique())
graphics = st.selectbox('GRAPHICS', GRAPHICS_OPTIONS)
ram = st.selectbox('RAM(GB)', [16, 24, 36, 64, 2, 4, 8, 12, ])

# map options to numerical values
BRAND_MAP = {'ASUS': 1, 'LENOVO': 2, 'HP': 3, 'MSI': 4, 'ACER': 5, 'DELL': 6, 'APPLE': 7, 'OTHER': 8, 'AVITA': 9, 'GIGABYTE': 10}
MODEL_MAP = {'VIVOBOOK': 1, 'MACBOOK': 2, 'INSPIRON': 3, 'TUF': 4, 'ASPIRE': 6, 'ZENBOOK': 7, 'LEGION': 8, 'ROG': 9, 'EXPERTBOOK': 10, 'MODERN': 11, 'PAVILION': 12, 'PROBOOK': 13}
OS_MAP = {'WINDOWS': 1, 'LINAX/DOS': 2, 'MACOS': 3, 'OTHERS': 4}
DISPLAY_MAP = {'FHD': 1, 'NORMAL': 2, 'LED': 3, 'IPS': 4}
GRAPHICS_MAP = {'INTEL': 1, 'NVIDIA': 2, 'OTHERS': 3, 'AMD': 4}
TOUCH_MAP = {'YES': 1, 'NO': 2}
CPU_MAP = {'AMD': 1, 'INTEL I5': 2, 'INTEL I7': 2, 'INTEL I3': 2, 'OTHERS': 2}


# button to trigger prediction
if st.button('Predict Price'):

    # replace the value of the processor
    cpu = CPU_MAP.get(cpu, 2)
    brand = BRAND_MAP.get(brand, 8)  # 8 is the default value if brand is not found
    model = MODEL_MAP.get(model, 5)  # 5 is the default value if model is not found
    os = OS_MAP.get(os, 4)           # 4 is the default value if os is not found
    display = DISPLAY_MAP.get(display, 2)  # 2 is the default value if display is not found
    graphics = GRAPHICS_MAP.get(graphics, 3)  # 3 is the default value if graphics is not found
    touch = TOUCH_MAP.get(touch, 2)  # 2 is the default value if touch is not found


    # prepare input data for prediction
    query = np.array([brand, model, cpu, display, touch, ssd, weight, os, adapter, battery, graphics, ram])
    query = query.reshape(1, -1)  # Reshape for prediction
    
    # perform prediction using the model
    predicted_price = xgb_model.predict(query)
    
    # display the predicted price
    st.write(f'Predicted Price: {predicted_price[0]}')








import streamlit as st 
import os
import joblib




import pandas as pd 
import numpy as np 


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


buying_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
maint_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
doors_label = {'2': 0, '3': 1, '5more': 2, '4': 3}
persons_label = {'2': 0, '4': 1, 'more': 2}
lug_boot_label = {'small': 0, 'big': 1, 'med': 2}
safety_label = {'high': 0, 'med': 1, 'low': 2}
class_label = {'good': 0, 'acceptable': 1, 'very good': 2, 'unacceptable': 3}


def get_value(val,my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
        

def get_key(val,my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
        
        
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model


def main():
    """Car ML App"""
    
    st.title("Car Evaluation ML App")
    st.subheader("Built with Streamlit")
    
    
    menu = ["EDA", "Prediction", "About"]
    choices = st.sidebar.selectbox("Select Activities",menu)

    if choices == 'EDA':
        st.subheader("EDA")
        
        data = load_data('data/car_eval_dataset.csv')
        st.dataframe(data.head(10))
        
        if st.checkbox("Show Summary"):
            st.write(data.describe())
            
        if st.checkbox("Show Shape"):
            st.write(data.shape)
            
        if st.checkbox("Value Count Plot"):
            fig, ax = plt.subplots()
            st.write(data['class'].value_counts().plot(kind='bar', ax=ax))
            st.pyplot(fig)
            
        if st.checkbox("Pie Chart"):
            fig, ax = plt.subplots()
            st.write(data['class'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax))
            st.pyplot(fig)
        
    
    
    elif choices == "Prediction":
        st.subheader("Prediction")
        
        buying  = st.selectbox("Select Buying Level",tuple(buying_label.keys()))
        maint = st.selectbox("Select Maintenance Level",tuple(maint_label.keys()))
        doors = st.selectbox("Select Number of Doors",tuple(doors_label.keys()))
        persons = st.number_input("Select Number of persons",2,10)
        lug_boot = st.selectbox("Select Lug Boot",tuple(lug_boot_label.keys()))
        safety = st.selectbox("Select Safety Level",tuple(safety_label.keys()))
        
        v_buying = get_value(buying,buying_label)
        v_maint = get_value(maint,maint_label)
        v_doors = get_value(doors,doors_label)
        v_lug_boot = get_value(lug_boot,lug_boot_label)
        v_safety = get_value(safety,safety_label)
        
        pretty_data = {
            "buying":buying,
            "maint":maint,
            "doors":doors,
            "persons":persons,
            "lug_boot":lug_boot,
            "safety":safety
        }
        
        st.subheader("Options Selected")
        st.json(pretty_data)
        
        st.subheader("Data Encoded As")
        sample_data = [v_buying,v_maint,v_doors,persons,v_lug_boot,v_safety]
        st.write(sample_data)
        
        prep_data = np.array(sample_data).reshape(1,-1)
        
        model_choice = st.selectbox("Model Choice",["LogisticRegression","Naive Bayes", "MLP Classifier"])
        if st.button("Evaluate"):
            if model_choice == 'LogisticRegression':
                predictor = load_prediction_model("models/logit_car_model.pkl")
                prediction = predictor.predict(prep_data)
                
                
            elif model_choice == 'Naive Bayes':
                predictor = load_prediction_model("models/nb_car_model.pkl")
                prediction = predictor.predict(prep_data)
                
                
            elif model_choice == 'MLP Classifier':
                predictor = load_prediction_model("models/nn_clf_car_model.pkl")
                prediction = predictor.predict(prep_data)
                
                
            
            final_result = get_key(prediction,class_label)
            st.success(final_result)
    
    
    elif choices == "About":
        st.subheader("About")    
    
    
if __name__ == '__main__':
    main()
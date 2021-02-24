import streamlit as st
import pandas as pd
import altair as alt
import urllib
import re
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from io import StringIO
import locale

locale.setlocale(locale.LC_ALL, 'en_IN')

#DATA PREPROCESSING 
def load_and_preprocess(file_name):
    car_data = pd.read_csv(file_name)
    car_data.dropna(inplace=True)
    car_data.reset_index(inplace=True)
    print('NA droped')
    car_data.mileage = car_data.mileage.apply(lambda x:''.join(re.findall(r"[-+]?\d*\.+|\d+",x)))
    car_data.engine = car_data.engine.apply(lambda x:''.join(re.findall(r"[-+]?\d*\.+|\d+",x)))
    car_data.max_power = car_data.max_power.apply(lambda x:''.join(re.findall(r"[-+]?\d*\.+|\d+",x)))
    car_data.torque = car_data.torque.apply(lambda x:''.join(re.findall(r"[-+]?\d*\.+|\d+",x.split("Nm@")[0])))
    encoder_name = OneHotEncoder()
    encoded_car_names=pd.DataFrame(encoder_name.fit_transform(car_data[['name']]).toarray())
    encoder_feul = OneHotEncoder()
    feul_type = pd.DataFrame(encoder_feul.fit_transform(car_data[['fuel']]).toarray())
    seller_encoder = OneHotEncoder()
    seller_type =pd.DataFrame(seller_encoder.fit_transform(car_data[['seller_type']]).toarray())
    trans_encoder = OneHotEncoder()
    trans =  pd.DataFrame(trans_encoder.fit_transform(car_data[['transmission']]).toarray())
    owner_encoder = OneHotEncoder()
    owner =  pd.DataFrame(owner_encoder.fit_transform(car_data[['owner']]).toarray())
    fdf = pd.concat([car_data[['year', 'km_driven','mileage','engine','max_power','torque']],encoded_car_names,feul_type,seller_type,trans,owner],axis=1)
    orignal_res = car_data['selling_price']
    print('"FEATURES = [0] , EXPECTED_OUTPUT = [1]", helper_df=[2]')
    helper_df = {'encoder_name':encoder_name,'encoder_feul':encoder_feul,'seller_encoder':seller_encoder,'trans_encoder':trans_encoder,'owner_encoder':owner_encoder}
    return [fdf,orignal_res,helper_df]

def model(FEATURES,OUTPUTS,helper_df):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(FEATURES, OUTPUTS)
    model = [regressor,helper_df]
    pickle.dump(model,open( "model.pkl", "wb" ))
    return regressor

def prediction(regressor,input_df,helper_df):
    col_names = input_df.columns
    input_df.mileage = input_df.mileage.apply(lambda x:''.join(re.findall(r"[-+]?\d*\.+|\d+",x)))
    input_df.engine = input_df.engine.apply(lambda x:''.join(re.findall(r"[-+]?\d*\.+|\d+",x)))
    input_df.max_power = input_df.max_power.apply(lambda x:''.join(re.findall(r"[-+]?\d*\.+|\d+",x)))
    input_df.torque = input_df.torque.apply(lambda x:''.join(re.findall(r"[-+]?\d*\.+|\d+",x.split("Nm@")[0])))
    fdf = pd.concat( 
                    [input_df[['year', 'km_driven','mileage','engine','max_power','torque']],
                    pd.DataFrame(helper_df["encoder_name"].transform([input_df.name]).toarray()), 
                    pd.DataFrame(helper_df["encoder_feul"].transform([input_df.fuel]).toarray()),
                    pd.DataFrame(helper_df["seller_encoder"].transform([input_df.seller_type]).toarray()),
                    pd.DataFrame(helper_df["trans_encoder"].transform([input_df.transmission]).toarray()),
                    pd.DataFrame(helper_df["owner_encoder"].transform([input_df.owner]).toarray())],axis=1
                    )
    return  regressor.predict(fdf)
    

def give_me_price(data=None,train=False,file_name='Car details v3.csv'):
        input_ = data
        if train:
            try:
                FEATURES,EXPECTED_OUTPUT,helper_df=load_and_preprocess(file_name)
                regressor = model(FEATURES,EXPECTED_OUTPUT,helper_df)
                return True
            except Exception as e:
                return e
        else:
            regressor,helper_df = pd.read_pickle(r'model.pkl')
        return str(prediction(regressor,input_,helper_df)[0])

@st.cache
def get_UN_data():
    df = pd.read_csv("Car details v3.csv")
    df.dropna(inplace=True)

    return df
try:
  
    df = get_UN_data()
    cars = st.sidebar.selectbox(
        "Choose Car Model", list(sorted(df.name.unique(),reverse=False))
    )

    year = st.sidebar.selectbox(
        "Choose Year", list(sorted(df.year.unique(),reverse=True))
    )

    km_driven = st.sidebar.slider('Enter Kms', min_value=int(
        0), max_value=500000, step=int(1000))

    fuel = st.sidebar.selectbox('Fuel', df.fuel[df.name == cars].unique())

    seller_type = st.sidebar.selectbox('Seller Type', df.seller_type[df.name == cars].unique())

    transmission = st.sidebar.selectbox(
        'Transmission', df.transmission[df.name == cars].unique())

    owner = st.sidebar.selectbox('Owner', df.owner.unique())

    if not cars:
        st.error("Please select one model.")
    else:
        cars_df = df[df.name == cars].head(1)

        st.write("## Input DF to the model")

        columns = ['name', 'year', 'km_driven', 'fuel', 'seller_type',
           'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque',
           'seats']

        data = [[cars,int(year), km_driven, fuel, seller_type,transmission, owner,str(cars_df['mileage'].values[0]), cars_df['engine'].values[0], cars_df['max_power'].values[0], cars_df['torque'].values[0],cars_df['seats'].values[0]]]
        real_km = cars_df = df[df.name == cars]['km_driven'].mean()
        if km_driven==0.0:
            km_driven=1.0
        dif_per = real_km/km_driven
        new_df = pd.DataFrame(data,columns=columns)
        st.write(new_df.T)
        result = give_me_price(new_df)

        st.markdown(
            f"""
            ### Predicted Price for {cars}
            """)
        st.write(f"# {locale.format('%d', int(result.split('.')[0]), grouping=True)}. Rs.")
        if st.button('Retrain The model'):
              user_input = st.text_input("Please Enter File Name or skip",'Car details v3.csv')
              st.write('Retraining')
              ret = give_me_price(train=True,file_name=user_input)
              if ret==True:
                  st.write('Please Refresh The Page')
              else:
                  st.write('error : ',ret)
            

except urllib.error.URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )

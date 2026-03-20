# Importing essential libraries and modules

from flask import Flask, render_template, request
from markupsafe import Markup
import numpy as np
import pandas as pd
import seaborn as sns
import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import plotly.express as px
from models_details import multiple_models

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_dic= ['5','6', '7', '8','9']



#from model_predict  import pred_leaf_disease

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------
from import_analyse import basic_info,preprocess_data,eda_plots

# Assuming df is already loaded somewhere globally or within a function

#from recamandation_code import recondation_fn

app = Flask(__name__)


# Load the trained XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the mappings for categorical columns
with open('mappings.pkl', 'rb') as file:
    mappings = pickle.load(file)





@ app.route('/')
def home(): 
    title = 'Crop Yield Prediction Using Machine Learning'
    return render_template('index.html', title=title) 

# render crop recommendation form page
@app.route('/preprocessing_data')
def preprocessing_data():
    num_nulls_before, cat_nulls_before, num_nulls_after, cat_nulls_after, head_html = preprocess_data('output.csv')
    return render_template('preprocessing_page.html', num_nulls_before=num_nulls_before, cat_nulls_before=cat_nulls_before, num_nulls_after=num_nulls_after, cat_nulls_after=cat_nulls_after, head=head_html)


@app.route('/eda_data')
def eda_data():
    eda_plots('output.csv')  # Generate plots


    # Return the path to the images for embedding in eda_page.html
    return render_template('eda_page.html', numerical_dist_img='static/images/numerical_distribution.png',
                           categorical_counts_img='static/images/categorical_counts.png',
                           heatmap_img='static/images/heatmap.png')



@app.route('/eda_data2')
def eda_data2():
    # Load the dataset
    df = pd.read_csv('output.csv')
    
    # Drop any unwanted or unnecessary columns
    df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')


    df=df.head(2000)
    
    # Update column names according to your new dataset
    df.columns = ['State', 'District', 'Crop Year', 'Season', 'Crop Name', 
                  'Area (hectares)', 'Temperature (°C)', 'Wind Speed (m/s)', 
                  'Precipitation (mm)', 'Humidity (%)', 'Soil Type', 'Nitrogen (N)',
                  'Phosphorus (P)', 'Potassium (K)', 'Production (tons)', 'Pressure (hPa)']
    # Separate categorical and numerical columns
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    num_cols = [col for col in df.columns if df[col].dtype != 'object']



    
    # Record the number of null values before imputation
    num_nulls_before = df[num_cols].isnull().sum().to_dict()
    cat_nulls_before = df[cat_cols].isnull().sum().to_dict()
    
    # Function for random value imputation on numerical columns
    def random_value_imputation(feature):
        random_sample = df[feature].dropna().sample(df[feature].isna().sum())
        random_sample.index = df[df[feature].isnull()].index
        df.loc[df[feature].isnull(), feature] = random_sample

    # Function for imputing missing categorical values using mode
    def impute_mode(feature):
        mode = df[feature].mode()[0]
        df[feature] = df[feature].fillna(mode)
    
    # Apply random value imputation for all numerical columns
    for col in num_cols:
        random_value_imputation(col)
    
    # Apply mode imputation for all categorical columns
    for col in cat_cols:
        impute_mode(col)
    
    # Record the number of null values after imputation
    num_nulls_after = df[num_cols].isnull().sum().to_dict()
    cat_nulls_after = df[cat_cols].isnull().sum().to_dict() 
    # Convert 'Production (tons)' to numeric, coercing invalid values to NaN
    df["Production (tons)"] = pd.to_numeric(df["Production (tons)"], errors='coerce')

    # Check for any rows with NaN in 'Production (tons)'
    invalid_rows = df[df["Production (tons)"].isna()]

    # Option 1: Drop rows with NaN in 'Production (tons)'
    df.dropna(subset=["Production (tons)"], inplace=True) 
    # Plotting the violin plot for 'Production (tons)'
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, y='Production (tons)')
    plt.ylabel('Production (tons)')
    plt.savefig('static/images/production_violin_plot.png')  # Save the plot
    plt.close()

    # Plotting the scatter plot for 'Supply Volume (tons)' vs 'Production (tons)'
    plt.figure(figsize=(12, 8))
    sns.relplot(data=df, x='Area (hectares)', y='Production (tons)', kind='scatter')
    plt.xlabel('Area (hectares)')
    plt.ylabel('Production (tons)')
    plt.savefig('static/images/area_vs_production_scatter_plot.png')  # Save the plot
    plt.close()

    # Line plot for 'Supply Volume (tons)' vs 'Production (tons)'
    sns.relplot(data=df, x='Area (hectares)', y='Production (tons)', kind='line')
    plt.xlabel('Area (hectares)')
    plt.ylabel('Production (tons)')
    plt.savefig('static/images/area_vs_production_line_plot.png')  # Save the plot
    plt.close()

    # Line plot for 'Demand Volume (tons)' vs 'Production (tons)'
    sns.relplot(data=df, x='Temperature (°C)', y='Production (tons)', kind='line')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Production (tons)')
    plt.savefig('static/images/temperature_vs_production_line_plot.png')  # Save the plot
    plt.close()

    # Line plot for 'Rainfall (mm)' vs 'Production (tons)'
    sns.relplot(data=df, x='Precipitation (mm)', y='Production (tons)', kind='line')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Production (tons)')
    plt.savefig('static/images/rainfall_vs_production_line_plot.png')  # Save the plot
    plt.close()

    # Return the path to the images for embedding in the HTML template
    return render_template('eda_page2.html', 
                           numerical_dist_img='static/images/production_violin_plot.png',
                           categorical_counts_img='static/images/area_vs_production_scatter_plot.png',
                           heatmap_img='static/images/area_vs_production_line_plot.png',
                           heatmap_img2='static/images/temperature_vs_production_line_plot.png',
                           heatmap_img3='static/images/rainfall_vs_production_line_plot.png')



@app.route('/models_data')
def models_data():
    results=multiple_models('output.csv')
    #return render_template('index.html', results=results)
    return render_template('models_dt.html',results=results)

@app.route('/test_application')
def test_application():
    #return render_template('recommendation.html')
        # Pass the unique values for state, district, season, crop, and soil to the HTML form
    return render_template('recommendation.html', 
                           states=mappings['state_names'].keys(),
                           districts=mappings['district_names'].keys(),
                           seasons=mappings['season_names'].keys(),
                           crops=mappings['crop_names'].keys(),
                           soils=mappings['soil_type'].keys())



@app.route('/disease-predict2', methods=['GET', 'POST'])
def disease_prediction2():
    title = 'Crop Yield Prediction Using Machine Learning'
    return render_template('rust.html', title=title) 
 


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
            title = 'Crop Price Prediction Using Machinelearning'

    #if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)

            file = request.files.get('file')

           # if not file:
            #    return render_template('disease.html', title=title)

            #img = Image.open(file)
            file.save('output.csv')

#df.head(),df.shape,df.describe(),df.info()
            #df2=basic_info("output.csv")



            #table = df2.to_html(classes="table table-striped table-hover", border=0)
            head, shape, describe, info = basic_info('output.csv')
            return render_template('rust-result.html', head=head, shape=shape, describe=describe, info=info)
#prediction =pred_leaf_disease("output.BMP")

            #prediction = (str(disease_dic[prediction]))

           # print("print the blood group of the candidate ",prediction)

            #if prediction=="5":
            #        class_rust=5


            #elif prediction=="6":
            #        class_rust=6 


            #elif prediction=="7":
            #        class_rust=7



            #elif prediction=="8":
            #        class_rust=8



            #elif prediction=="9":
            #        class_rust="There is noe Corrosion"


           # return render_template('rust-result.html',table=table,title=title)
        #except:
         #   pass
    


# render disease prediction result page


with open('xgb2.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Now you can use `loaded_model` to make predictions



@app.route('/predict1',methods=['POST'])
def predict1():
     if request.method == 'POST':
                                try:
                                    # Get form data
                                    state_name = request.form['state_name']
                                    district_name = request.form['district_name']
                                    crop_year = int(request.form['crop_year'])
                                    season_name = request.form['season_name']
                                    crop_name = request.form['crop_name']
                                    area = float(request.form['area'])
                                    temperature = float(request.form['temperature'])
                                    wind_speed = float(request.form['wind_speed'])
                                    precipitation = float(request.form['precipitation'])
                                    humidity = float(request.form['humidity'])
                                    soil_type = request.form['soil_type']
                                    N = float(request.form['N'])
                                    P = float(request.form['P'])
                                    K = float(request.form['K'])
                                    pressure = float(request.form['pressure'])

                                    # Map object-type columns to their corresponding numeric values
                                    state_num = mappings['state_names'].get(state_name, -1)
                                    district_num = mappings['district_names'].get(district_name, -1)
                                    season_num = mappings['season_names'].get(season_name, -1)
                                    crop_num = mappings['crop_names'].get(crop_name, -1)
                                    soil_num = mappings['soil_type'].get(soil_type, -1)

                                    # Ensure all necessary values are provided
                                    if -1 in [state_num, district_num, season_num, crop_num, soil_num]:
                                        return "Invalid input. Please select valid options."

                                    # Prepare input for the model
                                    features = np.array([[state_num, district_num, crop_year, season_num, crop_num, area, temperature, wind_speed,
                                                          precipitation, humidity, soil_num, N, P, K, pressure]])

                                    # Make prediction
                                    prediction = model.predict(features)
                                    production_output = round(prediction[0], 2)

                                    # Pass the original input data and prediction to the HTML form
                                    return render_template('recommendation.html', 
                                                           prediction_text=f'Estimated Production: {production_output} tons',
                                                           states=mappings['state_names'].keys(),
                                                           districts=mappings['district_names'].keys(),
                                                           seasons=mappings['season_names'].keys(),
                                                           crops=mappings['crop_names'].keys(),
                                                           soils=mappings['soil_type'].keys())
                                
                                except Exception as e:
                                    return str(e)
       # return render_template('recommendation.html', prediction1_text='The Price of The crop Will Be  {} '.format(prediction[0]))


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)

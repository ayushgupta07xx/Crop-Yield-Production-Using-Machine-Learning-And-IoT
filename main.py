# main.py

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
import random, pickle, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')  # Add this line before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from import_analyse import basic_info, preprocess_data, eda_plots
from models_details import multiple_models

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sunharshini@gmail.com'  
app.config['MAIL_PASSWORD'] = 'atvw dxxe kfuz zcyb' # app password

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
mail = Mail(app)

# ================= User Model =================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    otp = db.Column(db.String(6), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= Forms =================
class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=150)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    otp = StringField('OTP (if received)')
    submit = SubmitField('Login')

# ================= Routes =================

@app.route('/')
@login_required
def home():
    return render_template('home.html', title="Crop Yield Prediction Using Machine Learning")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: 
        return redirect(url_for('home'))
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter((User.username == form.username.data) | (User.email == form.email.data)).first():
            flash('Username or email already exists.', 'danger')
        else:
            hashed_pw = generate_password_hash(form.password.data, method='pbkdf2:sha256')
            new_user = User(username=form.username.data, email=form.email.data, password=hashed_pw)
            db.session.add(new_user)
            db.session.commit()
            flash('Registered successfully. Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: 
        return redirect(url_for('home'))
    form = LoginForm()
    show_otp = False  # Flag to control OTP field visibility
    
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data, email=form.email.data).first()
        if user:
            if form.password.data and not form.otp.data:
                if not check_password_hash(user.password, form.password.data):
                    flash('Incorrect password.', 'danger')
                else:
                    # Generate OTP and send
                    user.otp = str(random.randint(100000, 999999))
                    db.session.commit()
                    try:
                        msg = Message('Your OTP Code', sender=app.config['MAIL_USERNAME'], recipients=[user.email])
                        msg.body = f"Your OTP is {user.otp}"
                        mail.send(msg)
                        flash('OTP sent. Enter OTP to complete login.', 'info')
                        show_otp = True  # Show OTP input on re-render
                    except Exception as e:
                        flash(f'OTP email failed: {e}', 'danger')
            elif form.otp.data and user.otp == form.otp.data:
                user.otp = None
                db.session.commit()
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid OTP.', 'danger')
                if user.otp:
                    show_otp = True  # Show OTP input again if OTP was sent previously
        else:
            flash('User not found.', 'danger')
    
    return render_template('login.html', form=form, show_otp=show_otp)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

# ================= Load Models and Mappings =================

from catboost import CatBoostRegressor

# Load CatBoost model
model = CatBoostRegressor()
model.load_model('catboost_model.cbm')

# Load label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

mappings = {col: dict(zip(le.classes_, le.transform(le.classes_)))
            for col, le in label_encoders.items()}
# ================= Prediction & Tools Routes =================

@app.route('/test_application')
@login_required
def test_application():
    return render_template(
        'recommendtrial.html',
        states=label_encoders['state_names'].classes_,
        districts=label_encoders['district_names'].classes_,
        seasons=label_encoders['season_names'].classes_,
        crops=label_encoders['crop_names'].classes_,
        soils=label_encoders['soil_type'].classes_
    )


@app.route('/predict1', methods=['POST'])
@login_required
def predict1():
    try:
        data = request.form

        # Encode categorical inputs using saved LabelEncoders
        state = label_encoders['state_names'].transform([data['state_name']])[0]
        district = label_encoders['district_names'].transform([data['district_name']])[0]
        season = label_encoders['season_names'].transform([data['season_name']])[0]
        crop = label_encoders['crop_names'].transform([data['crop_name']])[0]
        soil = label_encoders['soil_type'].transform([data['soil_type']])[0]

        # Build feature vector (order must match training columns)
        features = np.array([[
            state,
            district,
            int(data['crop_year']),
            season,
            crop,
            float(data['area']),
            float(data['temperature']),
            float(data['wind_speed']),
            float(data['precipitation']),
            float(data['humidity']),
            soil,
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['pressure'])
        ]])

        # Predict (trained with log transform → inverse here)
        pred_log = model.predict(features)[0]
        output = round(np.expm1(pred_log), 2)

        return render_template(
            'recommendtrial.html',
            prediction_text=f'Estimated Production: {output} tons',
            states=label_encoders['state_names'].classes_,
            districts=label_encoders['district_names'].classes_,
            seasons=label_encoders['season_names'].classes_,
            crops=label_encoders['crop_names'].classes_,
            soils=label_encoders['soil_type'].classes_
        )

    except Exception as e:
        return str(e)


@app.route('/preprocessing_data')
@login_required
def preprocessing_data():
    num_nulls_before, cat_nulls_before, num_nulls_after, cat_nulls_after, head_html = preprocess_data('output.csv')
    return render_template('preprocessing_page.html', num_nulls_before=num_nulls_before,
                           cat_nulls_before=cat_nulls_before, num_nulls_after=num_nulls_after,
                           cat_nulls_after=cat_nulls_after, head=head_html)

@app.route('/eda_data')
@login_required
def eda_data():
    eda_plots('output.csv')
    return render_template('eda_page.html',
        numerical_dist_img='static/images/numerical_distribution.png',
        categorical_counts_img='static/images/categorical_counts.png',
        heatmap_img='static/images/heatmap.png')

@app.route('/eda_data2')
@login_required
def eda_data2():
    # Add your EDA2 plotting code or call from import_analyse here if needed
    return render_template('eda_page2.html',
        numerical_dist_img='static/images/production_violin_plot.png',
        categorical_counts_img='static/images/area_vs_production_scatter_plot.png',
        heatmap_img='static/images/area_vs_production_line_plot.png',
        heatmap_img2='static/images/temperature_vs_production_line_plot.png',
        heatmap_img3='static/images/rainfall_vs_production_line_plot.png')


@app.route('/disease-predict2', methods=['GET', 'POST'])
def disease_prediction2():
    title = 'Crop Yield Prediction Using Machine Learning'
    return render_template('rust.html', title=title)

@app.route('/dataprep', methods=['GET', 'POST'])
def dataprep():
    title = 'Crop Yield Prediction Using Machine Learning'
    return render_template('rust.html', title=title)

@app.route('/models_data')
@login_required
def models_data():
    results = multiple_models('output.csv')
    return render_template('models_dt.html', results=results)

@app.route('/disease-predict', methods=['GET', 'POST'])
@login_required
def disease_prediction():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file.save('output.csv')
            head, shape, describe, info = basic_info('output.csv')
            return render_template('rust-result.html', head=head, shape=shape, describe=describe, info=info)
        else:
            flash("Please upload a valid file.", "warning")
            return redirect(request.url)
    return render_template('disease_predict.html')  # form to upload CSV

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

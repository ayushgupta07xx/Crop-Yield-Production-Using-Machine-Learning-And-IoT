from flask import Flask, render_template, redirect, url_for, flash, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from flask_mail import Mail, Message
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Or your preferred DB
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your-email-password'  # Replace with your email password or app password

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
mail = Mail(app)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)  # Store hashed passwords (see note below)
    otp = db.Column(db.String(6), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Forms
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
    otp = StringField('OTP (if you received one)')
    submit = SubmitField('Login')

# Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter((User.username==form.username.data)|(User.email==form.email.data)).first()
        if existing_user:
            flash('Username or email already exists.', 'danger')
            return render_template('register.html', form=form)
        # Hash password here (using werkzeug.security.generate_password_hash recommended)
        from werkzeug.security import generate_password_hash
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! You can now login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data, email=form.email.data).first()
        if user:
            from werkzeug.security import check_password_hash
            if form.password.data and not form.otp.data:
                # Check password
                if not check_password_hash(user.password, form.password.data):
                    flash('Invalid password', 'danger')
                    return render_template('login.html', form=form)
                # Generate and send OTP
                generated_otp = str(random.randint(100000, 999999))
                user.otp = generated_otp
                db.session.commit()
                # Send OTP email
                try:
                    msg = Message('Your OTP Code', sender=app.config['MAIL_USERNAME'], recipients=[user.email])
                    msg.body = f'Your OTP code is: {generated_otp}'
                    mail.send(msg)
                    flash('OTP sent to your email. Please enter the OTP to complete login.', 'info')
                except Exception as e:
                    flash(f'Failed to send OTP email: {str(e)}', 'danger')
                return render_template('login.html', form=form)
            elif form.password.data and form.otp.data:
                # Validate OTP
                if user.otp == form.otp.data:
                    user.otp = None  # clear OTP
                    db.session.commit()
                    login_user(user)
                    flash('Login successful!', 'success')
                    return redirect(url_for('home'))
                else:
                    flash('Invalid OTP', 'danger')
            else:
                flash('Password or OTP required', 'danger')
        else:
            flash('User does not exist', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html', title='Crop Yield Prediction Using Machine Learning')

if __name__ == '__main__':
    db.create_all()  # Create DB tables
    app.run(debug=True)

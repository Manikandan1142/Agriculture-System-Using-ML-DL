from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
import bcrypt
import cv2
from markupsafe import Markup
from networkx import generate_gml
import requests
import numpy as np
import pandas as pd
import pickle
import config
from PIL import Image
from datetime import datetime

import random
from datetime import datetime, timedelta



from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask import Flask, flash, json, redirect, render_template, url_for, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, Email, ValidationError
from wtforms import EmailField
from flask_mail import Mail, Message
from flask_bcrypt import Bcrypt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from utils.fertilizer import fertilizer_dic
from models import OTP  # Ensure this line is at the top of your file
from models import User,db
from forms import LoginForm, RegisterForm  # ✅ Import your RegisterForm

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

crop_recommendation_model_path = 'models/Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None



app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:1234@localhost/mani"
app.config["SECRET_KEY"] = 'thisissecretkey'

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "manivara22@gmail.com"  # Change this
app.config["MAIL_PASSWORD"] = "atxp ckjh bfdc tfyj"  # Change this

mail = Mail(app)


# ✅ Correct way to initialize the database
db.init_app(app)
# Initialize extensions
# db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


with app.app_context():
    db.create_all()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"



@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

        
def send_otp(recipient_email, otp_code):
    sender_email = "manivara22@gmail.com"  # Change this to your email
    sender_password = "atxp ckjh bfdc tfyj"  # Change this to your email password

    subject = "Your OTP Code"
    body = f"Your OTP code is: {otp_code}. Please enter this code to verify your account."

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print("OTP sent successfully!")
    except Exception as e:
        print("Failed to send OTP:", e)

class CropPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nitrogen = db.Column(db.Integer, nullable=False)
    phosphorous = db.Column(db.Integer, nullable=False)
    potassium = db.Column(db.Integer, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    state = db.Column(db.String(100), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    predicted_crop = db.Column(db.String(100), nullable=False)

# Contact Us Model
class ContactUs(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(500), nullable=False)
    text = db.Column(db.String(900), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.name}"

def generate_otp():
    return str(random.randint(100000, 999999))

# Load the trained pest classification model
MODEL_PATH = "models/pest_model.h5"
model = load_model(MODEL_PATH, compile=False)

# Load pesticide information
with open("Data/pesticides.json", "r", encoding="utf-8") as f:
    pesticide_info = json.load(f)

# Load OpenCV's pre-trained model for human detection
PROTO_PATH = "models/deploy.prototxt"
MODEL_WEIGHTS = "models/mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_WEIGHTS)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure upload folder exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to detect humans in the image
def detect_human(img_path):
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If confidence is greater than 50%, it's likely a human
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Class ID 15 corresponds to 'person'
                return True  # Human detected

    return False  # No human detected

# Function to process and predict the pest
def predict_pest(img_path):
    # Check if human is present
    if detect_human(img_path):
        return "Human Detected", {"error": "Please upload a pest image, not a human image."}

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_labels = list(pesticide_info.keys())

    if len(prediction[0]) != len(class_labels):
        return "Unknown Pest", {}

    class_index = np.argmax(prediction[0])
    detected_pest = class_labels[class_index]

    # Fetch pesticide information
    pest_data = pesticide_info.get(detected_pest, {"error": "Pesticide information not found"})

    return detected_pest, pest_data  # Return tuple (pest_name, pest_data)



    
@app.route("/")
def hello_world():
    return render_template("index.html")
    

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method=='POST':
        name = request.form['name']
        email = request.form['email']
        text = request.form['text']
        contacts = ContactUs(name=name, email=email, text=text)
        db.session.add(contacts)
        db.session.commit()
    
    return render_template("contact.html")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)  # ✅ This logs in the user
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))  # ✅ Redirect to dashboard

        else:
            flash("Invalid email or password.", "danger")
    
    return render_template("login.html", form=form)

@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    return render_template("dashboard.html", title="Dashboard", user=current_user)


@ app.route('/logout',methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('hello_world'))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = RegisterForm()

    if request.method == "POST":
        if "otp_code" in request.form:  # User entering OTP
            entered_otp = request.form["otp_code"]
            email = session["temp_user"]["email"]
            otp_entry = OTP.query.filter_by(email=email).first()

            if otp_entry and otp_entry.otp_code == entered_otp:
                hashed_password = bcrypt.generate_password_hash(session["temp_user"]["password"]).decode("utf-8")
                new_user = User(
                    full_name=session["temp_user"]["full_name"],
                    email=email,
                    password=hashed_password,
                    phone=session["temp_user"]["phone"],
                    is_verified=True
                )
                db.session.add(new_user)
                db.session.commit()

                # Clear session and OTP entry
                session.pop("temp_user", None)
                db.session.delete(otp_entry)
                db.session.commit()

                flash("Account created successfully! Please log in.", "success")
                return redirect(url_for("login"))
            else:
                flash("Invalid OTP. Please try again.", "danger")
                return render_template("signup.html", form=form, otp_sent=True)

        else:  # User submitting signup form
            email = form.email.data
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash("Email already registered. Please log in.", "danger")
                return redirect(url_for("login"))

            # Store user details temporarily
            session["temp_user"] = {
                "full_name": form.full_name.data,
                "email": email,
                "password": form.password.data,
                "phone": form.phone.data
            }

            # Generate OTP
            otp_code = str(random.randint(100000, 999999))

            # Store OTP in the database
            otp_entry = OTP.query.filter_by(email=email).first()
            if otp_entry:
                otp_entry.otp_code = otp_code  # Update OTP if already exists
            else:
                otp_entry = OTP(email=email, otp_code=otp_code)
                db.session.add(otp_entry)
            db.session.commit()

            # Send OTP via email
            send_otp(email, otp_code)  # Make sure this function works

            flash(f"OTP sent to {email}. Enter OTP to verify.", "info")
            return render_template("signup.html", form=form, otp_sent=True)

    return render_template("signup.html", form=form, otp_sent=False)





@ app.route('/crop-recommend')
@login_required
def crop_recommend():
    title = 'crop-recommend - Crop Recommendation'
    return render_template('crop.html', title=title)

@ app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    title = '- Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)
    
@app.route("/pesticide_recommendation", methods=["GET", "POST"])
def pesticide_recommendation():
    return render_template("pesticide_recommendation.html")





# render crop recommendation result page


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = '- Crop Recommendation'

    if request.method == 'POST':
        try:
            # Retrieve form data
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            state = request.form['stt']
            city = request.form['city']

            # Prepare data for prediction
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            # Store data in MySQL using SQLAlchemy
            new_entry = CropPrediction(
                nitrogen=N,
                phosphorous=P,
                potassium=K,
                ph=ph,
                rainfall=rainfall,
                temperature=temperature,
                humidity=humidity,
                state=state,
                city=city,
                predicted_crop=final_prediction
            )
            db.session.add(new_entry)
            db.session.commit()

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        except Exception as e:
            return f"Error: {e}"
        
# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = '- Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)


#Pesticides recommendation Result
@app.route("/pesticide_result", methods=["POST"])
def pesticide_result():
    if "file" not in request.files:
        return redirect(url_for("pesticide_recommendation"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("pesticide_recommendation"))

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    detected_pest, pest_data = predict_pest(file_path)  # Ensure `predict_pest` returns valid data

    return render_template(
        "pesticide_result.html",
        pest=detected_pest,
        pest_data=pest_data,
        image_path=file_path
    )

@app.route("/display")
def querydisplay():
    alltodo = ContactUs.query.all()
    return render_template("display.html",alltodo=alltodo)

@app.route("/AdminLogin", methods=['GET', 'POST'])
def AdminLogin():

    form = LoginForm()
    if current_user.is_authenticated:
         return redirect(url_for('admindashboard'))

    elif form.validate_on_submit():
        user = UserMixin.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password,form.password.data):
                login_user(user)
                return redirect(url_for('admindashboard'))

    return render_template("adminlogin.html", form=form)


    # return render_template("adminlogin.html")

@app.route("/admindashboard")
@login_required
def admindashboard():
    alltodo = ContactUs.query.all()
    alluser = User.query.all()
    return render_template("admindashboard.html",alltodo=alltodo, alluser=alluser)

@app.route("/reg",methods=['GET', 'POST'])
def reg():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = UserMixin(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('AdminLogin'))

    return render_template("reg.html", form=form)


if __name__ == "__main__":
    app.run(debug=True,port=8000)

import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from train_model import Flatten
from load_model import Prediction


basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+os.path.join(basedir, 'digits_db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
model = Prediction()


class Digit(db.Model):
    __tablename__ = 'handwriten_digits'
    id = db.Column(db.Integer, primary_key=True)
    digit_image = db.Column('digit_image', db.Text(1000000))
    prediction = db.Column('prediction', db.Text(1000000))
    guess = db.Column('good_bad', db.String(10))
    prob = db.Column('prob', db.Text(1000000))


def calculate_rate():
    try:
        total_guess = Digit.query.count()
        correct_guess = Digit.query.filter_by(guess='true').count()
        prct = round(correct_guess / total_guess, 2)*100
        return correct_guess, total_guess , prct
    except:
        correct_guess, total_guess, prct = 0, 0, 0
        return correct_guess, total_guess, prct


@app.route('/prediction', methods=['POST'])
def prediction():
    image = request.form['save_image']
    result, image_base64 = model.predict_digit(image)
    img = f'data:image/png;base64,{image_base64}'
    return jsonify(result, img)


@app.route('/', methods=['POST', 'GET'])
def enter_new():
    corr, total, rate = calculate_rate()
    if request.method == 'POST':
        image = request.form['save_image']
        predict = request.form['prediction']
        guess = request.form['guess']
        prob_img = request.form['prob_img']
        digit = Digit(digit_image=image, prediction=predict, guess=guess, prob=prob_img)
        db.session.add(digit)
        db.session.commit()
        return redirect(url_for('enter_new'))
    else:
        return render_template('index.html', correct_guess=corr, total_guess=total, prct=rate)


@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    all_digits = Digit.query.order_by(Digit.id.desc()).paginate(page=page, per_page=5)
    return render_template('history.html', digits=all_digits)


@app.route('/delete/<id>')
def delete(id):
    digit = Digit.query.get(id)
    db.session.delete(digit)
    db.session.commit()
    return redirect(url_for('history'))


if __name__ == '__main__':
    db.create_all()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)

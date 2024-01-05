from . import app
from .result import Result
from flask import (render_template, request, url_for,
                   jsonify, redirect)
from src.features import Features
import os
import joblib
from werkzeug.utils import secure_filename

import librosa as lb
#from .project.forms import MessageForm

@app.route('/', methods=['GET', 'POST'])
def file_upload():

    if request.method == 'POST':

        file = request.files.get('file') or None

        if file.filename == '' or (not file.filename.endswith('.wav')):
            return jsonify({'error': 'Please upload a WAV file'})

        max_file_size = 10 * 1024 * 1024
        # 10 MB (adjust this as needed)

        try:
            filename = secure_filename(file.filename)
            if filename != '':
                file_ext = os.path.splitext(filename)[1]

                file.save(os.path.join(app.root_path,
                                       app.config['UPLOAD_FOLDER'],
                                       "user.wav"))
                file.close()

                audio = (
                    os.path.join(os.path.join(app.root_path,
                                                  app.config['UPLOAD_FOLDER'],
                                              "user.wav")))

                file_size = os.path.getsize(audio)

                if (file_size > max_file_size):
                    return jsonify({'error': "File size exceeds the limit of 10 MB!"})

                y, sr = lb.load(audio, sr = None)

                os.remove(audio)

                num_mfcc = 13
                duration = 5
                features = Features( num_mfcc, duration, y, sr)

                knn_model_file = (
                    os.path.join(os.path.join(app.root_path,
                                              app.config['MODEL_FOLDER'],
                                              'knn_model.pkl')))

                knn_model = joblib.load(knn_model_file)

                scaler_file = (
                    os.path.join(os.path.join(app.root_path,
                                                app.config['MODEL_FOLDER'],
                                                'scaler.pkl')))

                scaler = joblib.load(scaler_file)
                predict = (
                    features.predict(features.collect_features(),
                                     knn_model, scaler))

                result = Result()
                result.result = predict[0]

            return redirect(url_for('report', predict = predict[0] ))

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('upload.html')

@app.route('/report/<predict>')
def report(predict):
   #result = Result()
   return render_template("report.html", result = predict)

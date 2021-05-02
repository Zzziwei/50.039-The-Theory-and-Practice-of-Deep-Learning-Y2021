from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
from inference import *
 
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, "video")
ALLOWED_EXTENSIONS = {"avi"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_no_audio():
    if request.method == 'POST':
        print("in post")
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        print(allowed_file(file.filename))
        if file and allowed_file(file.filename):
            print("inside hre")
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)

            video_label, video_prob = infer_video(filename)
            audio_label, audio_prob = infer_audio(filename)
 
            return render_template("index.html", filename=filename, video_label=video_label, video_prob=video_prob, audio_label=audio_label, audio_prob=audio_prob)

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
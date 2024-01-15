#-*- coding: utf-8 -*-
import os
from flask import (
    Flask,
    request,
    render_template,
    Response,
    url_for,
    redirect
)
from model import voiceRemove, mp4ToMp3

app = Flask(__name__)

UPLOAD_VIDEO_FOLDER = './static/video'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        upload_file = request.files['upload_file']
        video_path = os.path.join(UPLOAD_VIDEO_FOLDER, upload_file.filename)
        upload_file.save(video_path)
        voiceRemove(video_path, "input.mp3")
        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, send_from_directory
import os
import time
from video_processor import process_video

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
BG_VIDEO_DIR = '/Users/mantapa/VIDEOS BG'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def list_bg_videos():
    files = os.listdir(BG_VIDEO_DIR)
    return [f for f in files if f.lower( ).endswith(('.mp4', '.mov', '.avi'))]

@app.route('/', methods=['GET', 'POST'])
def index():
    video_ready = False
    elapsed_time = 0
    bg_videos = list_bg_videos()

    if request.method == 'POST':
        input_video = request.files['input_video']
        bg_filename = request.form['bg_choice']

        if not input_video or not bg_filename:
            return "Missing file(s)", 400

        input_path = os.path.join(UPLOAD_FOLDER, input_video.filename)
        bg_path = os.path.join(BG_VIDEO_DIR, bg_filename)
        output_path = os.path.join(OUTPUT_FOLDER, "output_final.mp4")

        input_video.save(input_path)

        start_time = time.time()
        process_video(input_path, bg_path, output_path)
        elapsed_time = round(time.time() - start_time, 2)

        os.remove(input_path)

        video_ready = True
        return render_template('index.html', video_ready=video_ready, timestamp=int(time.time()), elapsed_time=elapsed_time, bg_videos=bg_videos)

    return render_template('index.html', video_ready=False, bg_videos=bg_videos)

@app.route('/download_video')
def download_video():
    return send_from_directory(OUTPUT_FOLDER, "output_final.mp4", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
# Green Screen Video Processor

This Flask application lets you upload a portrait-oriented green-screen video and composite it over a chosen background video. The background retains its native resolution while the foreground subject is automatically scaled, positioned, and cleaned using chroma keying techniques.

## 🚀 Features

- Automatic green screen removal (YCrCb-based)
- Face-aware end trimming (cuts when the head leaves the frame)
- Green spill suppression (via HSV saturation control)
- Customizable person scaling and alignment
- Upload UI and background video selection
- Buffered frame processing for smoother output
- Output as MP4 (H.264)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone 
   cd greenscreen-video-app

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   Set your background video directory

3. **Edit this path in the Flask app:**


BG_VIDEO_DIR = '/Users/mantapa/VIDEOS BG'

Replace it with the directory where your background .mp4 files are stored.

## Project Structure

├── app.py                  # Flask web server
├── video_processor.py      # Core video processing logic
├── uploads/                # Temporary folder for uploaded videos
├── static/                 # Output video stored here
├── templates/
│   └── index.html          # Upload UI
├── requirements.txt
└── README.md

## 📌 Notes
Only .mp4, .mov, and .avi formats are supported.

The subject is detected and scaled to fit the background while preserving aspect ratio.

Output is saved as static/output_final.mp4.
from flask import Flask, render_template, Response, session
from flask_wtf import FlaskForm
import secrets
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
from infer import *

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

output_directory = 'yolo_assets/Uploads'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
app.config['UPLOAD_FOLDER'] = output_directory


class UploadFileForm(FlaskForm):
    """
    Represents the form to upload a video file for object detection.
    """
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(source=''):
    """
    Generator function that yields frames with object detection results from images or videos.

    Args:
        source (str): The source of the image or video frames.

    Yields:
        bytes: Frames with object detection results in JPEG format.
    """
    yolo_output = run_yolo(source=source, web_app=True)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_web():
    """
    Generator function that yields frames with object detection results from webcam.

    Yields:
        bytes: Frames with object detection results in JPEG format.
    """
    yolo_output = run_yolo(source=0, web_app=True)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    """
    Renders the home page.
    """
    session.clear()
    return render_template('index.html')




@app.route('/Detections', methods=['GET', 'POST'])
def upload_page():
    """
    Handles the media upload and saves the media file.
    """
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['media_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('predict1.html', form=form)


@app.route('/media')
def media():
    """
    Streams the media with object detection results.
    """
    media_path = session.get('media_path')
    return Response(generate_frames(source=media_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webapp')
def webapp():
    """
    Streams the video with object detection results from the webcam.
    """
    return Response(generate_frames_web(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == "__main__":
    app.run(debug=True)

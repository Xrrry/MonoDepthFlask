import os
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for
from flask import render_template, send_file, send_from_directory,json, jsonify, make_response
from test_simple import *

UPLOAD_FOLDER = './save'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# download_model_if_doesnt_exist(args.model_name)
model_path = os.path.join("models", "mono_640x192")
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)

depth_decoder.to(device)
depth_decoder.eval()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/feedback/<s>')
def feedback(s):
    return s


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def uploaded_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            left_sound, right_sound, sound_level = test_simple('.\\save\\' + filename, feed_width, feed_height, device, encoder, depth_decoder)
            return get_file(filename.split('.')[0] + '_disp.jpeg', left_sound, right_sound, sound_level)


@app.route('/get_file/<file_name>', methods=['GET'])
def get_file(file_name, left_sound, right_sound, sound_level):
    directory = './save'
    headers = {
        'Content-Type': 'image/jpeg',
        'File-Name': file_name,
        'Left-Volumn': left_sound,
        'Right-Volumn': right_sound,
        'Depth-Level': sound_level
    }
    try:
        response = make_response(
            send_from_directory(directory, file_name, as_attachment=True))
        response.headers = headers
        return response
    except Exception as e:
        return jsonify({"code": "异常", "message": "{}".format(e)})


if __name__ == '__main__':
    print("app")
    app.run()

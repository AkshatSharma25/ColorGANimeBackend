from flask import Flask, request, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from PIL import Image
import generator
import helper




app = Flask(__name__)
CORS(app)  # This enables CORS for all routes from any domain




UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


target_size = (128, 128)
transform = generator.transforms.Compose([
    generator.transforms.Resize((128, 128)),
    generator.transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])  # maps [0,1] → [-1,1]
])




checkpoint = generator.torch.load('model\checkpoint24.pth', weights_only=True)
generator.gen.load_state_dict(checkpoint['model_state_dict'])
# disc.load_state_dict(checkpoint['disc_state_dict'])
generator.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
# opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
epoch = checkpoint['epoch']  # Optional: use to resume from the correct epoch

generator.gen.eval()







@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    filename = secure_filename(file.filename)
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)
    new_image_path = os.path.join(UPLOAD_FOLDER, 'reconstructed_' + filename)
    helper.breakIntoPieces()
    # print("i got it")
    helper.colorise(generator.gen)
    helper.reconstruct()
    helper.delete_all_files_in_folder("uploads")
    helper.delete_all_files_in_folder("fragments")
    new_image_path="return.png"
    return send_file(
        new_image_path, 
        mimetype='image/png', 
        as_attachment=True,
        download_name='result.png'
    )
    # return "hello world"

# import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)


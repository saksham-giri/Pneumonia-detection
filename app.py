```python
from flask import Flask, render_template, request, redirect
import os

# Disable GPU checks (helps on Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once at startup
model_path = os.path.join('model', 'pneumonia_model.h5')
model = load_model(model_path)


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150, 150))

    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                filename
            )

            file.save(filepath)

            try:
                processed_image = preprocess_image(filepath)

                prediction = model.predict(processed_image)

                result = (
                    "Pneumonia detected"
                    if prediction[0][0] > 0.5
                    else "No Pneumonia detected"
                )

            except Exception as e:
                result = f"Prediction error: {str(e)}"

            # Optional cleanup
            if os.path.exists(filepath):
                os.remove(filepath)

            return render_template(
                'index.html',
                prediction=result
            )

    return render_template(
        'index.html',
        prediction=None
    )


if __name__ == '__main__':
    app.run(debug=True)
```

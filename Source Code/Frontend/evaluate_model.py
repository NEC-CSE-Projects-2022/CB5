from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about/about.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart/flowchart.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics/metrics.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('File uploaded successfully!')
            return redirect(url_for('train', filename=filename))
        else:
            flash('Invalid file format! Only .csv and .xlsx allowed.')
            return redirect(request.url)
    return render_template('prediction/base.html')

@app.route('/train/<filename>', methods=['GET'])
def train(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        from model.train_model import train_model
        results = train_model(filepath, target_column='Label')  
        return render_template('prediction/results.html', results=results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file or file.filename == '':
                return jsonify({"error": "No file provided"}), 400
            if not allowed_file(file.filename):
                return jsonify({"error": "Invalid file format"}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            from model.predict_model import predict_model
            predictions = predict_model(filepath)  
            return jsonify({"predictions": predictions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return render_template('prediction/base.html')

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple API test endpoint."""
    return jsonify({"message": "API is working"}), 200

if __name__ == '__main__':
    app.run(debug=True)

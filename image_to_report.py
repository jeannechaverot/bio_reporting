from flask import Flask, request, jsonify, send_file
import cv2
import pandas as pd
import numpy as np

app = Flask(__name__)

# Define a route for the root URL ("/")
@app.route('/')
def index():
    return "Welcome to the Image to Excel App!"

@app.route('/process_image', methods=['POST'])
def process_image():
    # Receive the uploaded image
    image = request.files['image']

    # Process the image (fake processing)
    processed_data = np.random.rand(5, 5)

    # Convert the processed data to a pandas DataFrame
    df = pd.DataFrame(processed_data, columns=['Column1', 'Column2', 'Column3', 'Column4', 'Column5'])

    # Save the DataFrame as an Excel file
    excel_filename = 'processed_data.xlsx'
    df.to_excel(excel_filename, index=False)

    # Return the Excel file for download
    return send_file(excel_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

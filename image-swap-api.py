from flask import Flask, request, jsonify
import os
import subprocess
import requests
import boto3
from botocore.exceptions import NoCredentialsError
import uuid

app = Flask(__name__)

# Define paths
base_path = "/home/azureuser/facefusion/"
source_path = os.path.join(base_path, "source.jpg")
target_path = os.path.join(base_path, "target.jpg")
output_path = os.path.join(base_path, "output.jpg")
script_path = os.path.join(base_path, "facefusion.py")

# AWS configuration from environment variables
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
aws_region = os.getenv("AWS_REGION")

# Function to download images from URLs
def download_image(url, file_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

# Function to upload file to S3
def upload_to_s3(file_path, bucket_name):
    try:
        # Generate a unique object name using UUID
        unique_filename = f"{uuid.uuid4()}.jpg"
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        s3_client.upload_file(file_path, bucket_name, unique_filename)
        s3_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{unique_filename}"
        return s3_url
    except NoCredentialsError:
        raise Exception("AWS credentials not found")
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {str(e)}")

@app.route('/faceswap', methods=['POST'])
def face_swap():
    try:
        data = request.json
        source_url = data.get('source_url')
        target_url = data.get('target_url')

        if not source_url or not target_url:
            return jsonify({"error": "Source and target URLs are required"}), 400

        # Download images
        if not download_image(source_url, source_path):
            return jsonify({"error": "Failed to download source image"}), 500
        if not download_image(target_url, target_path):
            return jsonify({"error": "Failed to download target image"}), 500

        # Construct the command
        command = [
            "python3", script_path, "headless-run",
            "--source-paths", source_path,
            "--target-path", target_path,
            "--output-path", output_path,
            "--processor", "face_swapper",
            "--face-detector-model", "yoloface",
            "--face-detector-size", "640x640",
            "--face-detector-angles", "0", "90", "180", "270",
            "--face-detector-score", "0.5",
            "--face-landmarker-model", "2dfan4",
            "--face-landmarker-score", "0.5",
            "--face-selector-mode", "reference",
            "--face-selector-order", "large-small",
            "--face-selector-gender", "male",
            "--face-selector-age-start", "0",
            "--face-selector-age-end", "100",
            "--reference-face-distance", "0.6",
            "--face-mask-types", "box", "region",
            "--face-mask-blur", "0.3",
            "--face-mask-padding", "0", "0", "0", "0",
            "--execution-providers", "cpu",
            "--execution-thread-count", "4",
            "--execution-queue-count", "1",
            "--face-swapper-pixel-boost", "256x256",
            "--output-image-quality", "100",
            "--output-image-resolution", "1920x1080",
            "--log-level", "info"
        ]

        # Run the script
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode != 0:
            return jsonify({"error": "Facefusion script failed", "details": process.stderr}), 500

        # Upload the output image to S3 with a unique filename
        output_s3_url = upload_to_s3(output_path, s3_bucket_name)

        return jsonify({"message": "Face swap completed successfully", "output_s3_url": output_s3_url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860)

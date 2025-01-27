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
source_path_1 = os.path.join(base_path, "source1.jpg")
source_path_2 = os.path.join(base_path, "source2.jpg")
target_path = os.path.join(base_path, "target.jpg")
output_path = os.path.join(base_path, "output.jpg")
secondary_output_path = os.path.join(base_path, "output_secondary.jpg")
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
        source_url_1 = data.get('source_url_1')
        source_url_2 = data.get('source_url_2')
        target_url = data.get('target_url')

        if not source_url_1 or not source_url_2 or not target_url:
            return jsonify({"error": "Two source URLs and a target URL are required"}), 400

        # Download images
        if not download_image(source_url_1, source_path_1):
            return jsonify({"error": "Failed to download first source image"}), 500
        if not download_image(source_url_2, source_path_2):
            return jsonify({"error": "Failed to download second source image"}), 500
        if not download_image(target_url, target_path):
            return jsonify({"error": "Failed to download target image"}), 500

        # First run with "large-small" and the first source image
        command_first_run = [
            "python3", script_path, "headless-run",
            "--source-paths", source_path_1,
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

        process_first = subprocess.run(command_first_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process_first.returncode != 0:
            return jsonify({"error": "Facefusion script failed on first run", "details": process_first.stderr}), 500

        # Second run with "small-large" and the second source image, using the first output as the target
        command_second_run = [
            "python3", script_path, "headless-run",
            "--source-paths", source_path_2,
            "--target-path", output_path,  # Output of the first run is now the target
            "--output-path", secondary_output_path,
            "--processor", "face_swapper",
            "--face-detector-model", "yoloface",
            "--face-detector-size", "640x640",
            "--face-detector-angles", "0", "90", "180", "270",
            "--face-detector-score", "0.5",
            "--face-landmarker-model", "2dfan4",
            "--face-landmarker-score", "0.5",
            "--face-selector-mode", "reference",
            "--face-selector-order", "small-large",
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

        process_second = subprocess.run(command_second_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process_second.returncode != 0:
            return jsonify({"error": "Facefusion script failed on second run", "details": process_second.stderr}), 500

        # Upload both outputs to S3
        first_output_s3_url = upload_to_s3(output_path, s3_bucket_name)
        second_output_s3_url = upload_to_s3(secondary_output_path, s3_bucket_name)

        return jsonify({
            "message": "Face swap completed successfully",
            "first_output_s3_url": first_output_s3_url,
            "second_output_s3_url": second_output_s3_url
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6099)

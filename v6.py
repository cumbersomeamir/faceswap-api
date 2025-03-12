from flask import Flask, request, jsonify
import os
import random
import subprocess
import requests
import boto3
from botocore.exceptions import NoCredentialsError
import uuid

app = Flask(__name__)

# Define paths
base_path = "/home/azureuser/facefusion/"
faceswap_images_path = os.path.join(base_path, "faceswap-images")
source_path = os.path.join(base_path, "source.jpg")
target_path_template = os.path.join(base_path, "target_{}.jpg")
output_path_template = os.path.join(base_path, "output_{}.jpg")
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

# Function to select random target images based on gender
def select_target_images(gender, num_images):
    gender_folder = os.path.join(faceswap_images_path, gender)
    if not os.path.exists(gender_folder):
        raise Exception(f"No folder found for gender: {gender}")
    
    images = [img for img in os.listdir(gender_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(images) < num_images:
        raise Exception(f"Not enough images in {gender_folder} for {num_images} face swaps")
    
    selected_images = random.sample(images, num_images)  # Randomly select non-repeating images
    return [os.path.join(gender_folder, img) for img in selected_images]

@app.route('/five-images-faceswap', methods=['POST'])
def face_swap():
    try:
        data = request.json
        source_url = data.get('source_url')
        gender = data.get('gender')

        if not source_url or not gender:
            return jsonify({"error": "Source URL and gender are required"}), 400

        # Download source image
        if not download_image(source_url, source_path):
            return jsonify({"error": "Failed to download source image"}), 500

        # Select 3 unique target images based on gender
        try:
            selected_target_images = select_target_images(gender, 5)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        output_s3_urls = []

        # Perform face swaps for each target image
        for i, target_image_path in enumerate(selected_target_images):
            try:
                # Define paths for the current iteration
                current_target_path = target_path_template.format(i + 1)
                current_output_path = output_path_template.format(i + 1)

                # Copy the target image
                os.system(f"cp '{target_image_path}' '{current_target_path}'")

                # Construct the command
                command = [
                    "python3", script_path, "headless-run",
                    "--source-paths", source_path,
                    "--target-path", current_target_path,
                    "--output-path", current_output_path,
                    "--processor", "face_swapper",
                    "--face-detector-model", "yoloface",
                    "--face-detector-size", "640x640",
                    "--face-detector-angles", "0", "90", "180", "270",
                    "--face-detector-score", "0.5",
                    "--face-landmarker-model", "2dfan4",
                    "--face-landmarker-score", "0.5",
                    "--face-selector-mode", "reference",
                    "--face-selector-order", "large-small",
                    "--face-selector-gender", gender,
                    "--face-selector-age-start", "0",
                    "--face-selector-age-end", "100",
                    "--reference-face-distance", "0.6",
                    "--face-mask-types", "box", "region",
                    "--face-mask-blur", "0.3",
                    "--face-mask-padding", "0", "0", "0", "0",
                    "--execution-providers", "cpu",
                    "--execution-thread-count", "4",
                    "--execution-queue-count", "1",
                    "--output-image-quality", "100",
                    "--output-image-resolution", "1920x1080",
                    "--log-level", "info",
                    "--skip-download"
                ]

                # Run the script
                process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if process.returncode != 0:
                    return jsonify({"error": f"Facefusion script failed for target image {i+1}", "details": process.stderr}), 500

                # Upload the output image to S3
                output_s3_url = upload_to_s3(current_output_path, s3_bucket_name)
                output_s3_urls.append(output_s3_url)

            except Exception as e:
                return jsonify({"error": f"Failed on image {i+1}: {str(e)}"}), 500

        return jsonify({"message": "Face swaps completed successfully", "output_s3_urls": output_s3_urls}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8013)

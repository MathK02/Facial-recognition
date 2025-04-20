import os
from PIL import Image

def compress_and_resize_images(input_dir, output_dir, target_size=(100, 100), quality=50):
    """
    Compress and resize images to reduce file size and dimensions.

    Args:
        input_dir (str): Path to the directory containing original images.
        output_dir (str): Path to save compressed and resized images.
        target_size (tuple): Target size for resizing (width, height).
        quality (int): Quality of the output image (1-100).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                # Construct the full input and output file paths
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path, file)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    # Open the image
                    img = Image.open(input_path).convert('RGB')

                    # Resize the image using LANCZOS resampling
                    img = img.resize(target_size, Image.Resampling.LANCZOS)

                    # Save the image with compression
                    img.save(output_path, "JPEG", quality=quality)

                    print(f"Compressed and resized: {output_path}")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    # Define paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "DataSetCreated")  # Input folder with images
    output_dir = os.path.join(script_dir, "CompressedDataSetCreated")  # Output folder for processed images

    # Parameters for resizing and compression
    target_size = (200, 200)  # Resize images to 100x100 pixels
    quality = 100 #% de qualité, 100 = on touche à rien

    # Run the compression and resizing function
    compress_and_resize_images(input_dir, output_dir, target_size, quality)

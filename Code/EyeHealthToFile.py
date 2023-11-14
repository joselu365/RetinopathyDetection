import csv
import os
from PIL import Image

def flip_images_in_directory(directory, flipped_path):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter files containing "right_flipped"
    right_flipped_files = [file for file in files if "right_flipped" in file]
    left_files = [file for file in files if "left" in file]

    # Copy left image
    for file in left_files:
        file_path = os.path.join(directory, file)
        try:
            # Open the image
            image = Image.open(file_path)

            # Save the image to the flipped_path
            flipped_file_path = os.path.join(flipped_path, file)  # Fix: Added a separator ","
            image.save(flipped_file_path)

            print(f"Image {file} saved to {flipped_file_path}")

        except Exception as e:
            print(f"Error processing image {file}: {str(e)}")

    # Process each right_flipped image
    for file in right_flipped_files:
        file_path = os.path.join(directory, file)
        try:
            # Open the image
            image = Image.open(file_path)

            # Flip the image horizontally
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Save the flipped image
            flipped_file_path = os.path.join(flipped_path, file)  # Fix: Added a separator ","
            flipped_image.save(flipped_file_path)

            print(f"Image {file} flipped and saved as {flipped_file_path}")

        except Exception as e:
            print(f"Error processing image {file}: {str(e)}")


def rename_files(csv_file_path, folder_path):
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header row if it exists
        next(csv_reader, None)
        
        for row in csv_reader:
            if len(row) >= 3:
                file_name_with_extension = row[0]
                side = row[1]
                health = row[2]
                
                # Extract base name and file extension
                base_name, file_extension = os.path.splitext(file_name_with_extension)
                
                # Construct old and new file paths
                old_path = os.path.join(folder_path, file_name_with_extension)
                if side == "1":
                    new_name = f"{base_name}_{health}_right_flipped{file_extension}"
                    # Flip the image and save it with the new name
                    #new_path = os.path.join(folder_path, new_name)
                    #flip_image(old_path, new_path)
                    #print(f"Flipped and renamed: {old_path} -> {new_path}")
                if side == "0":
                    new_name = f"{base_name}_{health}_left{file_extension}"
                    
                new_path = os.path.join(folder_path, new_name)
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    # Replace 'csv_file_path' , 'folder_path' and "flipped_path" with own paths
    csv_file_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/0original/drLabels.csv'
    folder_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/0original/'
    flipped_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/1Flipped/'

    # Uncomment to dump health and side to filename
    # rename_files(csv_file_path, folder_path)
    # Uncomment to flip
    # flip_images_in_directory(folder_path, flipped_path)




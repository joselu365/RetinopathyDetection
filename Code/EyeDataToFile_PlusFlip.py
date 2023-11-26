""" ///////////////////////////////////////////////////////////////////////////////
//                   Radt Eye
// Date:         14/11/2023
//
// File: EyeDataToFile_PlusFlip.py
// Description: Take data from csv file and adds extra info to file name, 
//              in addition flip the right eyes to match the same direction.
/////////////////////////////////////////////////////////////////////////////// """

import csv
import os
import shutil
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
                if side == "0":
                    new_name = f"{base_name}_{health}_left{file_extension}"
                    
                new_path = os.path.join(folder_path, new_name)
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")



def organize_files(folder_path, organized_path):
    # Create subfolders based on the second number
    subfolders = set()

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Extract the second number from the file name
            print(filename)
            second_number = filename.split('_')[1]
            second_number
            # Create a subfolder for each unique second number
            subfolder_path = os.path.join(organized_path, second_number)
            subfolder_path
            if second_number not in subfolders:
                os.makedirs(subfolder_path)
                subfolders.add(second_number)

            # Move the file to the corresponding subfolder
            shutil.copy(os.path.join(folder_path, filename), os.path.join(subfolder_path, filename))

# Replace 'input.csv' and 'folder_path' with the actual CSV file and folder paths
csv_file_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/0original/drLabels.csv'
folder_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/0original/'
flipped_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/1Flipped/'
organized_path = 'C:/Users/JoseLu/Desktop/Fundus_dataflow/Database/2Organized/'

#rename_files(csv_file_path, folder_path)
# Call the function to flip right_flipped images in the specified directory
#flip_images_in_directory(folder_path, flipped_path)

organize_files(flipped_path, organized_path)
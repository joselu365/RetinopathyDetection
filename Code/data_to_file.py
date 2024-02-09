"""
///////////////////////////////////////////////////////////////////////////////
//  Project: Development of an Automated Diabetic Retinopathy
//                   Detection System Based on Deep Learning Techniques
// Date:         14/11/2023
//
// File: data_to_file.py
// Description: Takes data from a CSV file, renames files in a specified folder
//              based on the CSV data.
///////////////////////////////////////////////////////////////////////////////
"""

import csv
import os
import shutil

def rename_and_process_files(csv_file_path, source_folder_path, target_folder_path):
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # Skip the header

        for row in csv_reader:
            if len(row) >= 3:
                original_file_name = row[0]
                health_status = row[2]
                base_name, file_extension = os.path.splitext(original_file_name)

                old_path = os.path.join(source_folder_path, original_file_name)
                new_file_name = f"{base_name}_{health_status}{file_extension}"
                new_path = os.path.join(target_folder_path, new_file_name)

                if os.path.exists(old_path):
                    # Optionally process the file (e.g., flip image) before moving
                    # flip_image(old_path)  # Uncomment and implement if needed
                    
                    # Ensure the target folder exists
                    os.makedirs(target_folder_path, exist_ok=True)
                    
                    # Move and rename the file to organize it
                    shutil.move(old_path, new_path)
                    print(f"Renamed and moved: {old_path} -> {new_path}")
                else:
                    print(f"File not found: {old_path}, cannot rename.")

# Update these paths according to your actual file locations
csv_file_path = 'Database/original/drLabels.csv'
source_folder_path = 'Database/original/'
target_folder_path = 'Database/2Organized/'

rename_and_process_files(csv_file_path, source_folder_path, target_folder_path)

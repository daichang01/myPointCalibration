import os
import re

def rename_files_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith("Image_") and filename.endswith(".bmp"):
            #分割文件提取数字部分
            parts = filename.split("_")
            number_part = parts[1].split(".")[0]
            if number_part.isdigit():
                number = int(number_part)
                if number % 2 == 0:
                    new_number = number - 1
                    new_filename = f"Image_{new_number}.bmp"
                    old_file_path = os.path.join(folder_path, filename)
                    new_file_path = os.path.join(folder_path, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f'renamed "{filename}" to "{new_filename}"')

if __name__ == "__main__":
    rename_files_in_folder("dataset14/right")
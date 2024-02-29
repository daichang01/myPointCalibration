import os
import re
import cv2

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
                    new_number = number + 1
                    new_filename = f"Image_{new_number}.bmp"
                    old_file_path = os.path.join(folder_path, filename)
                    new_file_path = os.path.join(folder_path, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f'renamed "{filename}" to "{new_filename}"')

def split_stereo_iamges(source_folder, dest_folder_left, dest_folder_right):
    #确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_folder_left):
        os.makedirs(dest_folder_left)
    if not os.path.exists(dest_folder_right):
        os.makedirs(dest_folder_right)
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(source_folder, filename)
            img = cv2.imread(img_path)
            print(img.shape)

            #分割图片
            height, width = img.shape[:2]
            split_width = width // 2
            left_img = img[:,:split_width]
            right_img = img[:,split_width:]

            #构建目标文件路径并保存图片
            left_img_path = os.path.join(dest_folder_left, filename)
            right_img_path = os.path.join(dest_folder_right, filename)
            cv2.imwrite(left_img_path, left_img)
            cv2.imwrite(right_img_path, right_img)
    
def split_stereo_image():
    img = cv2.imread("testyichen.png")
    height, width = img.shape[:2]
    split_width = width // 2
    left_img = img[:,:split_width]
    right_img = img[:,split_width:]
    cv2.imwrite("testleft.png",left_img)
    cv2.imwrite("testright.png",right_img)



if __name__ == "__main__":
    rename_files_in_folder("Hkvs_dataset/left")
    # split_stereo_iamges("dataset/ori","dataset/left", "dataset/right")

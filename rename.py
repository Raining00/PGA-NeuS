import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".JPG"):
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, filename.lower())
            os.rename(old_path, new_path)

# 指定你的目录路径
directory = "./public_data/box/image"
rename_files(directory)

import zipfile
import sys
import os

if __name__ == "__main__":
    path_to_zip_file = sys.argv[1]
    name_file = path_to_zip_file.split('.')[0]

    os.makedirs(name_file, exist_ok=True)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('./{}/'.format(name_file))

    os.remove(path_to_zip_file)


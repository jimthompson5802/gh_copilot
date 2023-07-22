import argparse
import os
import pandas as pd

def get_python_module_files(directory):
    python_module_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / 1024
                python_module_files.append({'file': file_path, 'size_kb': file_size})
    return python_module_files

def main():
    # command line argument to get directory name
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory to get python module files from")
    args = parser.parse_args()

    # Set your directory here
    target_directory = args.directory

    python_module_files = get_python_module_files(target_directory)
    df = pd.DataFrame(python_module_files)
    print(df.head())

    print("File Size Distribution of Python modules:")
    print(df['size_kb'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

if __name__ == "__main__":
    main()

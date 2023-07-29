import os


class FileLoader:
    @staticmethod
    def load_file(file_name: str) -> str:
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File '{file_name}' not found.")
        
        with open(file_name, "r") as file:
            return file.read()

import ast

def extract_functions_and_classes(file_path):
    with open(file_path, 'r') as file:
        source_code = file.read()

    tree = ast.parse(source_code)

    functions_and_classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            function_source = ast.get_source_segment(source_code, node)
            functions_and_classes.append((function_name, function_source))

        if isinstance(node, ast.ClassDef):
            class_name = node.name
            class_source = ast.get_source_segment(source_code, node)
            functions_and_classes.append((class_name, class_source))

    return functions_and_classes

if __name__ == "__main__":
    python_file_path = "multivariate_ols.py"
    extracted_functions_and_classes = extract_functions_and_classes(python_file_path)

    for name, source in extracted_functions_and_classes:
        print(f"Name: {name}")
        print(f"Source code:\n{source}\n{'=' * 50}")

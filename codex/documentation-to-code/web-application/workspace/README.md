Based on the requirements, here are the core classes, functions, and methods that will be necessary:

1. `FileLoader` class: Responsible for loading the specified file and returning its contents.
   - `load_file(file_name: str) -> str`: Loads the specified file and returns its contents as a string.

2. `WebPage` class: Responsible for creating and rendering the web page.
   - `render(file_contents: str) -> str`: Renders the web page with the specified file contents.

3. `App` class: The main application class that brings everything together.
   - `run()`: Starts the application and listens for incoming connections.

Now let's proceed with creating the necessary files and their content.

1. `file_loader.py`


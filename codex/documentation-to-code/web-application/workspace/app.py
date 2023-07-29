from file_loader import FileLoader
from web_page import app


class App:
    def run(self):
        app.run(host="0.0.0.0", port=6000)


if __name__ == "__main__":
    App().run()

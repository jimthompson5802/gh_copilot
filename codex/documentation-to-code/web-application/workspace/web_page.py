from flask import Flask, render_template

app = Flask(__name__)


class WebPage:
    @staticmethod
    def render(file_contents: str) -> str:
        return render_template("index.html", file_contents=file_contents)


@app.route("/")
def index():
    file_contents = ""  # Placeholder for file contents
    return WebPage.render(file_contents)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)

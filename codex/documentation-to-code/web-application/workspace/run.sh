python3 -m venv myenv

source myenv/bin/activate

pip install flask

export FLASK_APP=app.py
flask run

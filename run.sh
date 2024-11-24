#!/bin/bash
DEVELOPMENT=true

# Log block
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

source venv/bin/activate
pip freeze > requirements.txt

if [ "$DEVELOPMENT" = true ]; then
    echo "Running Flask in development mode..."
    export FLASK_APP=app.py
    export FLASK_ENV=development
    flask run --host=0.0.0.0 --port=5000
else
    echo "Running Flask with Gunicorn in production mode..."
    gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
fi

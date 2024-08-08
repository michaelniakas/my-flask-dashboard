#!/bin/bash

# Start Gunicorn processes
echo Starting Gunicorn.
exec gunicorn --workers 4 --bind 0.0.0.0:$PORT app:app


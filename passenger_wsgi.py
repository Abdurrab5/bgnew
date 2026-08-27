import os
import sys

# Project directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import FastAPI app
from main import app

# Convert ASGI -> WSGI for Passenger
from a2wsgi import ASGIMiddleware

application = ASGIMiddleware(app)
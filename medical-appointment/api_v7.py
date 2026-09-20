"""Temporary eval harness for attempt 11 (cross-encoder repair).

Copy of api.py's server wiring only, serving example_v7.predict on port
9059. Does NOT touch api.py / example.py. Delete on revert.
"""

import datetime
import logging
import time

import uvicorn
from fastapi import FastAPI

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from example_v7 import predict
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9059

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()


@app.post('/predict', response_model=ASRQuestionResponseDto)
def predict_endpoint(request: ASRQuestionRequestDto):
    """Answer every question about one conversation."""
    response = predict(request)

    validate_response(response, expected_count=len(request.questions))

    return response


@app.get('/api')
def hello():
    return {
        'service': 'medical-appointment-usecase-v7',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api_v7:app', host=HOST, port=PORT)

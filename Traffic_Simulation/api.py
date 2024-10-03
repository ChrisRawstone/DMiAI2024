import uvicorn
from fastapi import FastAPI
import datetime
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
from time import time, sleep

app = FastAPI()

# Log initialization message
logger.info("Latency Monitoring API initialized.")

@app.get('/api')
def hello():
    return {
        "service": "signal-latency-monitor",
        "uptime": '{}'.format(datetime.timedelta(seconds=time()))
    }

@app.get('/')
def index():
    return "Signal latency monitoring endpoint is running!"

# Global variables to track signal states and their changes
signal_change_times = {}
active_signals = []
signal_state_log = []

green = 0
@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    global signal_change_times, active_signals, signal_state_log, green

    # Decode request data
    signals = request.signals
    current_tick = request.simulation_ticks
    logger.info(f"Current tick: {current_tick}")
    
    next_signals = []
    
    green += 1
  
    if green < 40:
        for signal in signals:
            next_signals.append(SignalDto(name=signal.name, state="green"))
        for signal in signals:
            logger.info(f"Signal {signal.name} is {signal.state}")
    else: 
        for signal in signals:
            next_signals.append(SignalDto(name=signal.name, state="red"))
        for signal in signals:
            logger.info(f"Signal {signal.name} is {signal.state}")
            
    if green == 40:
        print("send all red")

    # Return the updated signals to the simulation
    response = TrafficSimulationPredictResponseDto(signals=next_signals)
    return response

# Log results to a file
def log_results_to_file():
    filename = "signal_latency_log.txt"
    with open(filename, "w") as f:
        for log_entry in signal_state_log:
            signal_name, state, latency = log_entry
            f.write(f"Signal {signal_name} turned {state} with latency: {latency:.2f} seconds.\n")



if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host="0.0.0.0",
        port=8080
    )

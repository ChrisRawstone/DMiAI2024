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

@app.post('/monitor_signals', response_model=TrafficSimulationPredictResponseDto)
def monitor_signals(request: TrafficSimulationPredictRequestDto):
    global signal_change_times, active_signals, signal_state_log

    # Decode request data
    signals = request.signals
    current_tick = request.simulation_ticks
    logger.info(f"Current tick: {current_tick}")
    
    # Initialize signal change time tracking
    if not signal_change_times:
        signal_change_times = {signal.name: {'last_change': None, 'state': signal.state} for signal in signals}

    # Check and log when signals actually turn green
    for signal in signals:
        signal_name = signal.name
        signal_state = signal.state

        # Check if the signal has turned green
        if signal_change_times[signal_name]['state'] == "red" and signal_state == "green":
            latency = time() - signal_change_times[signal_name]['last_change']
            logger.info(f"Signal {signal_name} turned green with latency: {latency:.2f} seconds.")
            signal_state_log.append((signal_name, "green", latency))

        # Check if the signal has turned red
        elif signal_change_times[signal_name]['state'] == "green" and signal_state == "red":
            latency = time() - signal_change_times[signal_name]['last_change']
            logger.info(f"Signal {signal_name} turned red with latency: {latency:.2f} seconds.")
            signal_state_log.append((signal_name, "red", latency))

        # Update signal state and change time
        if signal_change_times[signal_name]['state'] != signal_state:
            signal_change_times[signal_name]['last_change'] = time()
            signal_change_times[signal_name]['state'] = signal_state

    # Determine whether to send all signals green or red based on the current state
    send_all_green = all(signal_change_times[signal.name]['state'] == "red" for signal in signals)
    send_all_red = all(signal_change_times[signal.name]['state'] == "green" for signal in signals)

    next_signals = []
    if send_all_green:
        logger.info("Sending all signals to green.")
        for signal in signals:
            next_signals.append(SignalDto(name=signal.name, state="green"))
            signal_change_times[signal.name]['last_change'] = time()  # Log the time we send the green signal

    elif send_all_red:
        logger.info("Sending all signals to red.")
        for signal in signals:
            next_signals.append(SignalDto(name=signal.name, state="red"))
            signal_change_times[signal.name]['last_change'] = time()  # Log the time we send the red signal

    else:
        logger.info("Waiting for all signals to change before sending new commands.")

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

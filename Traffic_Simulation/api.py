import uvicorn
from fastapi import FastAPI
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto

HOST = "0.0.0.0"
PORT = 8080

app = FastAPI()
start_time = time.time()

@app.get('/api')
def hello():
    return {
        "service": "traffic-simulation-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

# For logging
current_signal_phase = 0  # To track which signal is green
change_duration = 5  # Duration for each signal phase in seconds
last_change_time = time.time()  # Time when the last change occurred

# Initialize duration tracking
signal_durations = None
first_iteration = True  # Flag to check if it's the first iteration

@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    
    # Decode request
    data = request
    vehicles = data.vehicles
    total_score = data.total_score
    simulation_ticks = data.simulation_ticks
    signals = data.signals
    signal_groups = data.signal_groups
    legs = data.legs
    allowed_green_signal_combinations = data.allowed_green_signal_combinations
    is_terminated = data.is_terminated
    
    if first_iteration:
        # Create lists and dictionaries in one go
        signals = [signal.name for signal in signals]
        states = {signal.name: signal.state for signal in signals}
        
        signal_durations = {signal: 0 for signal in signals}
        
        # {signal: (state, duration)}
        signal_state_duration = {signal: (states[signal], signal_durations[signal]) for signal in signals}
        first_iteration = False
    
    # update signal state and duration without updating the lists
    for i in range(len(signals)):
        signal_state_duration[signals[i]] = (signals[i].state, signal_durations[signals[i]]+time.time()-last_change_time)
        if signal_state_duration[signals[i]][0] == 'green': #resetting time if signal is green
            signal_state_duration[signals[i]] = (signal_state_duration[signals[i]][0], 0)
            
        
    logger.info(f'signal_state_duration: {signal_state_duration}')
    logger.info(f'Number of vehicles at tick {simulation_ticks}: {len(vehicles)}')
    if time() - last_change_time > change_duration:
        current_signal_phase = (current_signal_phase + 1) % len(signals)  # Cycle through signals
        last_change_time = time.time()
    
    red_signal_group = []
    for idx, signal in enumerate(signals):
        if idx == current_signal_phase:
            green_signal_group = signal_groups[idx]
        else:
            red_signal_group = signal_groups.append(signal_groups[idx])
    

    # Select a signal group to go green
    # green_signal_group = signal_groups[0]

    

    # Return the encoded image to the validation/evalution service
    response = TrafficSimulationPredictResponseDto(
        signals=[SignalDto(
            name=green_signal_group,
            state="green"
        ), SignalDto(
            name=red_signal_group,
            state="red"
        )],
        
    )

    return response

if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
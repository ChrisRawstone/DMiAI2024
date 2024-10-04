import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto

app = FastAPI()

# Log initialization message
logger.info("Latency Monitoring API initialized.")

@app.get('/api')
def hello():
    return {
        "service": "signal-latency-monitor",
    }

@app.get('/')
def index():
    return "Signal latency monitoring endpoint is running!"

# Global variables to track signal states and their changes
global previous_states, shift_time, shift_to
previous_states = None
shift_time = 0
shift_to = None


@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    global previous_states, shift_time, shift_to
    next_signals = []
    # Decode request data
    signals = request.signals
    
    amber_time = 2
    red_amber_time = 4

    if previous_states is not None and shift_to is not None:
        if shift_to == "green":
            if all([signal.state == "green" for signal in signals]):
                print(f"It took {request.simulation_ticks - shift_time - amber_time} ticks to change the signals to green")
                for signal in signals:
                    print(f"Signal {signal.name} is {signal.state}")
                shift_to = None

        if shift_to == "red":
            if all([signal.state == "red" for signal in signals]):
                print(f"It took {request.simulation_ticks - shift_time - red_amber_time} ticks to change the signals to red")
                for signal in signals:
                    print(f"Signal {signal.name} is {signal.state}")
                shift_to = None

    if request.simulation_ticks % 30 == 0:
        for signal in signals:
            next_signals.append(SignalDto(name=signal.name, state="red"))
        shift_time = request.simulation_ticks
        shift_to = "red"
        print(f"Changing signals to red at tick {request.simulation_ticks}")

    if request.simulation_ticks % 30 == 15:
        for signal in signals:
            next_signals.append(SignalDto(name=signal.name, state="green"))
        shift_time = request.simulation_ticks
        shift_to = "green"
        print(f"Changing signals to green at tick {request.simulation_ticks}")

    # Save previous signal states
    previous_states = [signal.state for signal in signals]

    # Return the updated signals to the simulation
    # print("next signals:  ", next_signals)
    response = TrafficSimulationPredictResponseDto(signals=next_signals)
    return response


if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host="0.0.0.0",
        port=8080
    )

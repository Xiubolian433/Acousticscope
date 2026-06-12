import numpy as np
import sounddevice as sd
from scipy.signal import resample
import time
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datetime import datetime

def resample_audio(audio_data, original_sample_rate, target_sample_rate):
    resampled_data = resample(audio_data, int(len(audio_data) * target_sample_rate / original_sample_rate))
    return resampled_data

def speech_to_text(model, processor, device, max_duration=60, silence_threshold=0.2, sampling_rate=44100):
    timeout = 10  # 10 seconds timeout
    startTime = time.time()  # Initialize start time
    audio = np.array([])  # Initialize audio array

    while True:
        # Record a small chunk of audio to check the sound level
        init_chunk = sd.rec(int(sampling_rate * 0.5), samplerate=sampling_rate, channels=1)
        sd.wait()
        
        # Check sound level against silence threshold
        if np.max(np.abs(init_chunk)) >= silence_threshold:          
            print("I am listening...")
            audio = init_chunk.squeeze()
            break
        else:
            print("No voice detected, still listening...")
            if time.time() - startTime > timeout:
                print("Timeout reached, listening quitting.")
                current_time = datetime.now()
                print("Current time:", current_time.strftime("%H:%M:%S"))
                return "Listening Timeout"  # Return a message indicating timeout

    start_time = time.time()

    while True:
        chunk = sd.rec(int(sampling_rate), samplerate=sampling_rate, channels=1)
        sd.wait()
        audio = np.append(audio, chunk.squeeze())

        if np.max(np.abs(chunk)) < silence_threshold:
            break

        if len(audio) / sampling_rate >= max_duration:
            break

    try:
        # Resample audio to 16000 Hz for the model
        resampled_output = resample_audio(audio, sampling_rate, 16000)
        resampled_output = torch.tensor(resampled_output, dtype=torch.float32).to(device)
        model.to(device)
        input_values = processor(resampled_output, return_tensors="pt").input_values.to(device)

        with torch.no_grad():
            logits = model(input_values)
            tensor_output = logits[0]
        predicted_ids = torch.argmax(tensor_output, dim=-1)
        output = processor.batch_decode(predicted_ids)[0]
        
    except Exception as e:
        output = f"Error during transcription: {str(e)}"

    return output

# Example usage
# You would need to initialize the model, processor, and device before calling this function.
# model = Wav2Vec2ForCTC.from_pretrained("your_model_name")
# processor = Wav2Vec2Processor.from_pretrained("your_processor_name")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Then call the function
# text = speech_to_text(model, processor, device)
# print(text)
def main():
    # Initialize the model and processor for speech recognition
    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"  # Example model name
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # Set the device for computation (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Call the speech_to_text function
    text = speech_to_text(model, processor, device)
    print("Recognized Speech:", text)

if __name__ == "__main__":
    main()
import csv
from datetime import datetime
import sounddevice as sd
import numpy as np
assert np            # avoid "imported but unused" message (W0611)
from scipy.signal import resample
import time
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os.path
import speech_recognition as sr
import openai
from TTS.api import TTS
import os
import pygame
import pandas as pd


print("=====================================")
get_API = os.environ['COQUI_STUDIO_TOKEN'] = "0C1mFDWzCCJIdOjkWQ4J4rMzUGd470FAPhqAm71wYqld0CNBjfNZ23EruFu2HYO3"
if get_API:
    print("Get API!")

print("=====================================")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

# Set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device:
    print("GPU Activate!")
print("=====================================")
# Init TTS
tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v1.1"
tts = TTS(tts_model_name).to(device)

# Mixed precision setup
# Enable automatic mixed precision
scaler = torch.cuda.amp.GradScaler()
if tts:
    print("Initialized model successfully! Model name--->", tts_model_name)
print("=====================================")

pygame.mixer.init()

model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

# # model_name ="codenamewei/speech-to-text"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)


# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
OPEN_INTENT = 'Hey Alexa,'

openai.api_key = "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"

source_file = '/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing/Total_negative_Furst_matched_test.csv'
target_file = '/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing/Test_result/Mexican_female.csv'
speaker_voice = "/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing/Test Voice/Mexican_female.wav"

def read_from_files():
    # Load CSV files into Pandas DataFrames
    source_df = pd.read_csv(source_file)
    target_df = pd.read_csv(target_file)
    
    # Build two dictionaries: invoke_name ---> skill_id, invoke_name ---> skill_name
    invoke_to_skill_id = dict(zip(source_df['invoke_name'], source_df['skill']))
    invoke_to_skill_name = dict(zip(source_df['invoke_name'], source_df['skill_name']))
    
    return source_df, target_df, invoke_to_skill_id, invoke_to_skill_name


def write_to_file(target_df, invoke_to_skill_id, invoke_to_skill_name, speech, sentence, out, user_response,row):
  
  
    # Create a new DataFrame row with the new information
    new_data = {
        'skill_id': invoke_to_skill_id[row['invoke_name']],
        'skill_name': invoke_to_skill_name[row['invoke_name']],
        'All_invoke_words': speech,
        'invoke_name': sentence,
        'Alexa_Answer': out,
        'User_Response': user_response
    }
    
    # Append the new row to the DataFrame
    target_df = target_df.append(new_data, ignore_index=True)
  
    return target_df

def fileexist():
    if os.path.isfile(target_file):
        print("file exists")
    else:
        with open(target_file, mode='a+', newline='\n') as out:
            fieldnames = ['skill_id','skill_name','All_invoke_words', 'invoke_name','Alexa_Answer','User_Response']
            writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
            writer.writeheader()

def text_to_speech(sentence):
    text = sentence
    ##TODO: add loop to get the different voice from Test Voice folder
    tts.tts_to_file(text, speaker_wav=speaker_voice, language="en", file_path="output.wav",speed=0.6)
    file_path = "/home/mo/Documents/Alexa_Voice_Program/output.wav"
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    time.sleep(1)
    

def resample_audio(audio_data, original_sample_rate, target_sample_rate):
    resampled_data = resample(audio_data, int(len(audio_data) * target_sample_rate / original_sample_rate))
    return resampled_data


def speech_to_text(max_duration=60, silence_threshold=0.2, sampling_rate=44100):
    timeout = 10  # 30 seconds timeout
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

        if len(audio) / 44100 >= max_duration:
            text_to_speech("Hi, Alexa, stop")
            time.sleep(2)
            text_to_speech("Alexa, cancel")
            time.sleep(3)
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


def generate_response(out):
    # Set up OpenAI API request parameters
    prompt = f'''generate a fictional random response to this alexa response, be DIRECT, and be VERY CLEAR no more than 2 words, 
                if your answer contain "yes" the answer should be "yes" only no other words: DO NOT ANSWER WITH NO in the beginning{{{out}}}'''

    # Generate response using OpenAI API
    messages = [{"role": "user", "content": prompt}]

    retries = 3  # Set the number of retries
    while retries > 0:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
                max_tokens=70, #70-100
                n=1,
                stop=None,
                temperature=0.2,
            )
            response = response['choices'][0]['message']['content']
            return response  # Return the response if successful

        except openai.error.TimeoutError:
            print("Request timed out, retrying...")
            retries -= 1  # Decrement the retries counter
            time.sleep(10)  # Wait for 10 seconds before retrying

        except openai.error.OpenAIError as e:
            print(f"OpenAI Error: {e}")
            break  # Exit the loop on other OpenAI errors

    return "Unable to generate a response due to repeated timeouts or an error."


def main():
    source_df, target_df, invoke_to_skill_id, invoke_to_skill_name = read_from_files()

    for i,row in source_df.iterrows():
        speech = row['invoke_name']
        sentences = [s.strip('”\n') for s in speech.split('”,”')]

        sentence = sentences[0]
        text_to_speech("Hey alexa")
        text_to_speech(sentence)
        time.sleep(3)
        # Coversation with Alexa
        count = 0
        while count < 5:
            out = speech_to_text()
            #TODO: judge out if empty or not
            if out == '':
                out = "tag info,loop time: %d"%count
                user_response = "tag info: skill problem, skill stop"
                target_df = write_to_file(target_df, invoke_to_skill_id, invoke_to_skill_name, speech, sentence, out, user_response,row)
                target_df.to_csv(target_file, index=False)
                break

            user_response = generate_response(out)
            text_to_speech(user_response)
            target_df = write_to_file(target_df, invoke_to_skill_id, invoke_to_skill_name, speech, sentence, out, user_response,row)
            count = count +1
            # Save the updated DataFrame to a new CSV file
            target_df.to_csv(target_file, index=False)
           

        text_to_speech('Hey Alexa, Exit')
        time.sleep(2)
        text_to_speech('Hey Alexa, Cancel')
        current_time = datetime.now()
        print("Current time:", current_time.strftime("%H:%M:%S"))
        time.sleep(40)

if __name__ == "__main__":
    fileexist()
    main()

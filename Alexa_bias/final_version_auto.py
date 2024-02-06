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
import sys

source_file = '/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing/Total_negative_Furst_matched_test.csv'
voice_folder = '/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing/Test Voice'
target_folder = '/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing/Test_result'
key = "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"

def read_from_files(target_file):
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

def fileexist(test_results_folder,wav_filename):
    fieldnames = ['skill_id', 'skill_name', 'All_invoke_words', 'invoke_name', 'Alexa_Answer', 'User_Response']

    
    if not os.path.exists(test_results_folder):
        os.makedirs(test_results_folder)

    
    base_name = os.path.splitext(wav_filename)[0]
    csv_file = base_name + '.csv'  
    target_file = os.path.join(test_results_folder, csv_file)

  
    if not os.path.isfile(target_file):
        with open(target_file, mode='w', newline='\n') as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
            writer.writeheader()

    return target_file


def text_to_speech(model,speaker,sentence):
    pygame.mixer.init()  # Initialize the mixer
    model.tts_to_file(sentence, speaker_wav=speaker, language="en", file_path="output.wav",speed=0.6)
    file_path = "/home/mo/Documents/Alexa_Voice_Program/output.wav"
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    time.sleep(1)
    

def resample_audio(audio_data, original_sample_rate, target_sample_rate):
    resampled_data = resample(audio_data, int(len(audio_data) * target_sample_rate / original_sample_rate))
    return resampled_data


def speech_to_text(model, processor,tts,speaker_voice,device,max_duration=60, silence_threshold=0.2, sampling_rate=44100):
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
            text_to_speech(tts,speaker_voice,"Hi, Alexa, stop")
            time.sleep(5)
            text_to_speech(tts,speaker_voice,"Hi, Alexa, Exit")
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
    prompt = f'''generate a fictional random response to this alexa response, be DIRECT, and be VERY CLEAR no more than 5 words, 
                if your answer contain "yes" the answer should be "yes" and no more than 5 words: DO NOT ANSWER WITH NO in the beginning{{{out}}}'''

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

def sleep_countdown(seconds):
    for remaining in range(seconds, 0, -1):
        sys.stdout.write(f"\rSleep for {remaining} seconds...")
        sys.stdout.flush()
        time.sleep(1)
    print("\rSleep complete.")

def restart_countdown(seconds):
    for remaining in range(seconds, 0, -1):
        sys.stdout.write(f"\rRestart next voice for {remaining} seconds...")
        sys.stdout.flush()
        time.sleep(1)
    print("\rSleep complete.")    


def get_wav_files(test_voice_folder):
    return [f for f in os.listdir(test_voice_folder) if f.endswith('.wav')]

def main():
    test_voice_folder = voice_folder
    test_results_folder = target_folder
    wav_files = get_wav_files(test_voice_folder)
        
    for wav_file in wav_files:
        # check if csv files exist, if not, creat new one
        target_file = fileexist(test_results_folder,wav_file)
        # get the full path of csv
        target = os.path.join(test_results_folder, target_file)
        print(target)

        # get the full path of wav     
        wav = os.path.join(test_voice_folder, wav_file)

        # load voice model
        print("=====================================")
        get_API = os.environ['COQUI_STUDIO_TOKEN'] = "0C1mFDWzCCJIdOjkWQ4J4rMzUGd470FAPhqAm71wYqld0CNBjfNZ23EruFu2HYO3"
        if get_API:
            print("Get API!")
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

        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_name)

        openai.api_key = key
        
        speaker_voice = wav
        print("=====================================")
        print("use %s voice to start testing..." % wav_file)
        print("=====================================")

        source_df, target_df, invoke_to_skill_id, invoke_to_skill_name = read_from_files(target)

        for i,row in source_df.iterrows():
            speech = row['invoke_name']
            sentences = [s.strip('”\n') for s in speech.split('”,”')]

            sentence = sentences[0]
            text_to_speech(tts,speaker_voice,"Hey alexa")
            text_to_speech(tts,speaker_voice,sentence)
            time.sleep(3)
            # Coversation with Alexa
            count = 0
            while count < 7:
                out = speech_to_text(model, processor,tts,speaker_voice,device)
                #TODO: judge out if empty or not
                if out == '':
                    out = "tag info,loop time: %d"%count
                    user_response = "tag info: skill problem, skill stop"
                    target_df = write_to_file(target_df, invoke_to_skill_id, invoke_to_skill_name, speech, sentence, out, user_response,row)
                    target_df.to_csv(target_file, index=False)
                    break

                user_response = generate_response(out)
                text_to_speech(tts,speaker_voice,user_response)
                target_df = write_to_file(target_df, invoke_to_skill_id, invoke_to_skill_name, speech, sentence, out, user_response,row)
                count = count +1
                # Save the updated DataFrame to a new CSV file
                target_df.to_csv(target_file, index=False)


            text_to_speech(tts,speaker_voice,'Hey Alexa, Exit')
            time.sleep(2)
            text_to_speech(tts,speaker_voice,'Hey Alexa, Cancel')
            current_time = datetime.now()
            print("Current time:", current_time.strftime("%H:%M:%S"))
            print("Current file: %s" %wav_file)
            sleep_countdown(40)

        print("%s voice play complete!"%wav_file)
        # Start the sleep countdown
        restart_countdown(40)


if __name__ == "__main__":
    main()

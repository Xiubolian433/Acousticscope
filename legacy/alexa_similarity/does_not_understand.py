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
import threading 
import requests
import re
from gtts import gTTS
import subprocess






source_file = '<LOCAL_PROJECT_ROOT>/alexa_voice_testing/Alexa_Real copy.csv'
target_file = '<LOCAL_PROJECT_ROOT>/alexa_voice_testing/Alexa_Real_test copy.csv'

key = '<OPENAI_API_KEY>'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/wav2vec2-large-960h-lv60-self"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Move the model to GPU
    model = model.to("cuda")
    print("cuda available!")
else:
    print("CUDA is not available. The model will run on CPU.")    


def write_to_file(target_df,i,invoke_name, sentence, out, user_response):
  
  
    # Create a new DataFrame row with the new information
    new_data = {
        'number':i,
        'skill_name': invoke_name,
        'invoke_name': sentence,
        'Alexa_Answer': out,
        'User_Response': user_response
    }
    
    # Append the new row to the DataFrame
    target_df = target_df.append(new_data, ignore_index=True)
  
    return target_df

def ensure_csv_file_exists(target_file):
    fieldnames = ['skill_id', 'skill_name', 'invoke_name', 'Alexa_Answer', 'User_Response']

    # Check if the CSV file exists, create it with headers if it doesn't
    if not os.path.isfile(target_file):
        with open(target_file, mode='w', newline='\n') as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
            writer.writeheader()

def text_to_speech(sentence, output_file='output1.mp3'):
    tts = gTTS(text=sentence, lang='en', slow=True)
    tts.save(output_file)
    subprocess.run(['mpg321', output_file]) # Assuming you have mpg321 or another player installed

def text_to_speech_alexa(sentence,output_file='output2.mp3'):
    tts = gTTS(text=sentence, lang='en', slow=True)
    tts.save(output_file)
    subprocess.run(['mpg321', output_file]) # Assuming you have mpg321 or another player installed    


# def resample_audio(audio_data, original_sample_rate, target_sample_rate):
#     resampled_data = resample(audio_data, int(len(audio_data) * target_sample_rate / original_sample_rate))
#     return resampled_data















import librosa



import numpy as np
import sounddevice as sd
import torch
import time
from scipy.signal import resample

def resample_audio(audio_data, original_sample_rate, target_sample_rate):
    # More efficient resampling
    resampled_data = resample(audio_data, int(len(audio_data) * target_sample_rate / original_sample_rate))
    return resampled_data


def speech_to_text(model, processor, device, max_duration=120, silence_threshold=0.1, sampling_rate=44100):
    timeout = 10  # Timeout for initial voice detection
    start_time = time.time()
    audio_chunks = []  # Use a list to collect audio chunks
    silent_chunk_count = 0 
    max_silent_chunks = 4  # Increased to allow more silence before stopping
    record_duration = 0.25  # Slightly increased recording duration for each chunk

    # Start initial silent listening
    while time.time() - start_time < timeout:
        chunk = sd.rec(int(sampling_rate * record_duration), samplerate=sampling_rate, channels=1)
        sd.wait()
        chunk = chunk.squeeze()

        if np.max(np.abs(chunk)) >= silence_threshold:
            audio_chunks.append(chunk)
            print('I am listening...')
            break
        else:
            print("No voice detected, still listening...")

    if len(audio_chunks) == 0:
        return "Listening Timeout" 

    # Continue recording
    while True:
        chunk = sd.rec(int(sampling_rate * record_duration), samplerate=sampling_rate, channels=1)
        sd.wait()
        chunk = chunk.squeeze()
        audio_chunks.append(chunk)

        if np.max(np.abs(chunk)) < silence_threshold:
            silent_chunk_count += 1
            if silent_chunk_count >= max_silent_chunks:
                break
        else:
            silent_chunk_count = 0

        if sum([len(c) for c in audio_chunks]) / sampling_rate >= max_duration:
            break

    # Combine chunks and resample
    audio = np.concatenate(audio_chunks)
    try:
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
























# def speech_to_text(model, processor, device,max_duration=60, silence_threshold=0.2, sampling_rate=44100):
#     timeout = 10  # 30 seconds timeout
#     startTime = time.time()  # Initialize start time
#     audio = np.array([])  # Initialize audio arra
#     silent_chunk_count = 0 
#     max_silent_chunks = 4


#     while True:
#         # Record a small chunk of audio to check the sound level
#         init_chunk = sd.rec(int(sampling_rate * 0.5), samplerate=sampling_rate, channels=1)
#         sd.wait()
        
#         # Check sound level against silence threshold
#         if np.max(np.abs(init_chunk)) >= silence_threshold:          
#             print("I am listening...")
#             audio = init_chunk.squeeze()
#             break
#         else:
#             print("No voice detected, still listening...")
#             if time.time() - startTime > timeout:
#                 print("Timeout reached, listening quitting.")
#                 current_time = datetime.now()
#                 print("Current time:", current_time.strftime("%H:%M:%S"))
#                 return "Listening Timeout"  # Return a message indicating timeout

#     start_time = time.time()

#     while True:
#         chunk = sd.rec(int(sampling_rate), samplerate=sampling_rate, channels=1)
#         sd.wait()
#         audio = np.append(audio, chunk.squeeze())

#         if np.max(np.abs(chunk)) < silence_threshold:
#             silent_chunk_count += 1
#             if silent_chunk_count >= max_silent_chunks:
#                 break
#         else:
#             silent_chunk_count = 0

#         if len(audio) / 44100 >= max_duration:
#             text_to_speech("Hi, Alexa, stop")
#             time.sleep(5)
#             text_to_speech("Hi, Alexa, Exit")
#             time.sleep(3)
#             break 

#     try:
#         # Resample audio to 16000 Hz for the model
#         resampled_output = resample_audio(audio, sampling_rate, 16000)
#         resampled_output = torch.tensor(resampled_output, dtype=torch.float32).to(device)
#         model.to(device)
#         input_values = processor(resampled_output, return_tensors="pt").input_values.to(device)

#         with torch.no_grad():
#             logits = model(input_values)
#             tensor_output = logits[0]
#         predicted_ids = torch.argmax(tensor_output, dim=-1)
#         output = processor.batch_decode(predicted_ids)[0]
        
#     except Exception as e:
#         output = f"Error during transcription: {str(e)}"

#     return output

# class Chatbot:
#     def __init__(self, api_key):
#         openai.api_key = api_key
#         self.messages = ""  # Stores conversation history as a string
#         self.context = ""

#     def set_context(self, context):
#         self.context = context

#     def add_message(self, role, content):
#         prefix = "Q: " if role == "user" else "A: "
#         content = content.strip()
        
#         # Check if 'A: ' is already present for assistant responses
#         if role == "assistant" and content.startswith("A: "):
#             new_message = content + "\n\n"
#         else:
#             new_message = prefix + content + "\n\n"

#         self.messages += new_message

#         # Keep only the last 10 interactions (based on occurrences of "Q: ")
#         if self.messages.count("Q: ") > 10:
#             q_indices = [i for i in range(len(self.messages)) if self.messages.startswith("Q: ", i)]
#             self.messages = self.messages[q_indices[-10]:]

#     def send_message(self, message):
#         self.add_message("user", message)
#         max_retries = 3  # Set the maximum number of retries
#         retries = 0

#         while retries < max_retries:
#             try:
#                 response = openai.ChatCompletion.create(
#                     model="gpt-3.5-turbo",
#                     messages=[{"role": "system", "content": self.messages}],
#                     max_tokens=70,
#                     n=1,
#                     stop=None,
#                     temperature=0.7
#                 )
#                 response_message = response.choices[0].message["content"].strip()
#                 self.add_message("assistant", response_message)
                

#                 return response_message[3:]

#             except openai.error.OpenAIError as e:
#                 print(f"OpenAI API Error on attempt {retries + 1}: {e}")
#                 retries += 1
#                 time.sleep(10)  # Wait for a short period before retrying

#             except Exception as e:
#                 print(f"An unexpected error occurred on attempt {retries + 1}: {e}")
#                 retries += 1
#                 time.sleep(1)  # Wait for a short period before retrying

#         return 
    
#     def generate_response(self, out):
#         # prompt = f'''Generate a fictional, direct response to the following
#         #   text from Alexa. Your response should directly answer the given text(yes or no among other short answers),
#         #     be limited to no more than 3 words, make sure that the correct way to
#         #       answer is follow the exact instructions provided in the text 
#         #          Here is the text: {{{out}}}'''

#         prompt = f''' I want a concise response that directly answer the A pattern without explanations or 
#         adding a single word. You are B. finish the following conversation   """
# A:{{{out}}}
# B: 

# '''
#         response_message = self.send_message(prompt)
#         return response_message

class Chatbot:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.messages = []  # Stores conversation history as a list of strings
        self.context = ""

    def set_context(self, context):
        self.context = context

    def add_message(self, role, content):
        prefix = "Q: " if role == "user" else "A: "
        content = content.strip()
        # Check if 'A: ' is already present for assistant responses
        if role == "assistant" and content.startswith("A: "):
            new_message = content
        else:
            new_message = prefix + content

        self.messages.append(new_message)
        # Keep only the last 10 interactions
        self.messages = self.messages[-20:]

    def send_message(self, message):
        self.add_message("user", message)
        response_message = self.try_sending_message()
        if response_message:
            self.add_message("assistant", response_message)
        return response_message

    def try_sending_message(self):
        max_retries = 3
        for _ in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "\n".join(self.messages)}],
                    max_tokens=70,
                    n=1,
                    stop=None,
                    temperature=0.7
                )
                return response.choices[0].message["content"].strip()
            except openai.error.OpenAIError as e:
                print(f"OpenAI API Error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                time.sleep(1)
        return None

    def generate_response(self, out):
        prompt = f"I want a concise response that directly answers the pattern without explanations or adding extra words. You are B. Finish the following conversation:\n\nA: {out}\nB: "
        return self.send_message(prompt)

    
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





import time
   
def main():

    # Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device:
        print("GPU Activate!")
        print("=====================================")

    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Move the model to GPU
        model = model.to("cuda")
        print("cuda available!")
    else:
        print("CUDA is not available. The model will run on CPU.")    

    openai.api_key = key
    api_key = key
    chatbot = Chatbot(api_key)
    

    df = pd.read_csv(source_file)
    target_df = pd.DataFrame()

    for invoke_name in df['invoke_name']:
    
        for i in range(15):
            sentence = invoke_name
            print("[Hey, Alexa,]")
            text_to_speech("Hey Alexa,")
            time.sleep(1)
            print("[%s]"%sentence)
            text_to_speech(sentence)

            # Coversation with Alexa
            count = 0
            while count < 10:
                start_time = time.perf_counter()
                out = speech_to_text(model,processor, device)
                end_time = time.perf_counter()
                # Calculate the difference in time
                elapsed_time = end_time - start_time

                print(f"The function took {elapsed_time} seconds to run.")  
                
                
                #TODO: judge out if empty or not
                if out == '':
                    out = "tag info,loop time: %d"%count
                    user_response = "tag info: skill problem, skill stop"
                    target_df = write_to_file(target_df, i, invoke_name, sentence, out, user_response)
                    target_df.to_csv(target_file, index=False)
                    break

                if out == 'Listening Timeout' or len(out) == 3:
                    break

                # #Start the chatbot response generation in a separate thread
                # response_thread = threading.Thread(target=text_to_speech_alexa('Hi,Alexa,Alexa,Alexa,Alexa'))
                # response_thread.start()

                user_response = chatbot.generate_response(out)
           
                #Wait for the chatbot response generation to complete
                #response_thread.join()
                
                print(user_response)
                text_to_speech(user_response)
                time.sleep(2)
                target_df = write_to_file(target_df, i, invoke_name, sentence, out, user_response)
                count = count +1
                # Save the updated DataFrame to a new CSV file
                target_df.to_csv(target_file, index=False)

            print("Hey Alexa, Exit")
            text_to_speech('Hey Alexa, Exit')
            time.sleep(2)
            current_time = datetime.now()
            print("Current time:", current_time.strftime("%H:%M:%S"))
            sleep_countdown(15)

if __name__ == "__main__":
    ensure_csv_file_exists(target_file)
    main()
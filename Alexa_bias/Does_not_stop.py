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



source_file = '/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing_mo/Can not stop.csv'
voice_folder = '/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing_mo/Repire_file/Test_voice_repire'
target_folder = '/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing_mo/Repire_file/Test_result_repire'

key = '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'
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


# def text_to_speech(model,speaker,sentence):
#     pygame.mixer.init()  # Initialize the mixer
#     model.tts_to_file(sentence, speaker_wav=speaker, language="en", file_path="output.wav",speed=0.6)
#     file_path = "/home/mo/Documents/Alexa_Voice_Program/alexa_voice_testing_mo/output.wav"
#     pygame.mixer.music.load(file_path)
#     pygame.mixer.music.play()
#     time.sleep(1)

import subprocess
from gtts import gTTS


def text_to_speech(sentence, output_file='output1.mp3'):
    tts = gTTS(text=sentence, lang='en', slow=True)
    tts.save(output_file)
    subprocess.run(['mpg321', output_file])
 

def resample_audio(audio_data, original_sample_rate, target_sample_rate):
    resampled_data = resample(audio_data, int(len(audio_data) * target_sample_rate / original_sample_rate))
    return resampled_data



def speech_to_text(model, processor,tts,speaker_voice,device,max_duration=60, silence_threshold=0.2, sampling_rate=44100):
    timeout = 10  # 30 seconds timeout
    startTime = time.time()  # Initialize start time
    audio = np.array([])  # Initialize audio arra
    silent_chunk_count = 0 
    max_silent_chunks = 4


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
            silent_chunk_count += 1
            if silent_chunk_count >= max_silent_chunks:
                break
        else:
            silent_chunk_count = 0

        if len(audio) / 44100 >= max_duration:
            text_to_speech1(tts,speaker_voice,"Hi, Alexa, stop")
            time.sleep(5)
            text_to_speech1(tts,speaker_voice,"Hi, Alexa, Exit")
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



class Chatbot:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.messages = ""  # Stores conversation history as a string
        self.context = ""

    def set_context(self, context):
        self.context = context

    
    def clear_messages(self):
        self.messages = ""


    def add_message(self, role, content):
        prefix = "Q: " if role == "user" else "A: "
        content = content.strip()
        
        # Check if 'A: ' is already present for assistant responses
        if role == "assistant" and content.startswith("A: "):
            new_message = content + "\n\n"
        else:
            new_message = prefix + content + "\n\n"

        self.messages += new_message

        # Keep only the last 10 interactions (based on occurrences of "Q: ")
        if self.messages.count("Q: ") > 10:
            q_indices = [i for i in range(len(self.messages)) if self.messages.startswith("Q: ", i)]
            self.messages = self.messages[q_indices[-10]:]

    def send_message(self, message):
        self.add_message("user", message)
        max_retries = 3  # Set the maximum number of retries
        retries = 0

        while retries < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": self.messages}],
                    max_tokens=70,
                    n=1,
                    stop=None,
                    temperature=0.7
                )
                response_message = response.choices[0].message["content"].strip()
                if not response_message.startswith("A: "):
                    response_message = "A: " + response_message

                self.add_message("assistant", response_message)
                

                return response_message[3:]

            except openai.error.OpenAIError as e:
                print(f"OpenAI API Error on attempt {retries + 1}: {e}")
                retries += 1
                time.sleep(10)  # Wait for a short period before retrying

            except Exception as e:
                print(f"An unexpected error occurred on attempt {retries + 1}: {e}")
                retries += 1
                time.sleep(1)  # Wait for a short period before retrying

        return 
    
    def generate_response(self, out):
        # prompt = f'''Generate a fictional, direct response to the following
        #   text from Alexa. Your response should directly answer the given text(yes or no among other short answers),
        #     be limited to no more than 3 words, make sure that the correct way to
        #       answer is follow the exact instructions provided in the text 
        #          Here is the text: {{{out}}}'''

        prompt = f''' I want a concise response that directly answer the A pattern without explanations or 
        adding a single word. You are B. finish the following conversation   """
A:{{{out}}}
B: 

'''



        response_message = self.send_message(prompt)
        return response_message
  

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



def trigger_text_to_speech(event, tts, speaker_voice):
    if not event.is_set():
        text_to_speech(tts,speaker_voice,"Alexa, ")

def speech_to_text_wrapper(model, processor, tts, speaker_voice, device, results, event):
    result = speech_to_text(model, processor, tts, speaker_voice, device)
    results.insert(0, result)
    event.set()
    
def interact_with_skill(model, processor, device, tts, speaker_voice, chatbot, source_df, row):

    stt_completed = threading.Event()
    chatbot.clear_messages()
    
    results = []
    out=""
    user_response = ""
    speech = row['invoke_name']
    sentences = [s.strip('”\n') for s in speech.split('”,”')]

    sentence = sentences[0]
    

    
    text_to_speech("Hey Alexa, ")
    time.sleep(1)
    # text_to_speech1(tts,speaker_voice,sentence)
    text_to_speech(sentence)

    # Coversation with Alexa
    count = 0

    
    
    
    


    while count < 3:
        stt_thread = threading.Thread(target=speech_to_text_wrapper, args=(model, processor, tts, speaker_voice, device, results, stt_completed))
        stt_thread.start()

        tts_timer = threading.Timer(45, trigger_text_to_speech, args=(stt_completed, tts, speaker_voice))
        tts_timer.start()

        stt_thread.join()

        out = results[0] if results else "Error or no output from speech to text"
        tts_timer.cancel()

        if out == '' or out == 'Listening Timeout':
            break

        out = out.lower()
        user_response = chatbot.generate_response(out)
        text_to_speech(user_response)
        count += 1

def execute_command_and_record(command, model, processor, device, tts, speaker_voice, target_df, invoke_to_skill_id, invoke_to_skill_name, row, speech, sentence):
    count = 0
    while count < 2:
        time.sleep(2)
        text_to_speech(f'Alexa, {command}')
        
        out = speech_to_text(model, processor, tts, speaker_voice, device)
        out = out.lower()

        if out == '' or out == 'Listening Timeout':
            out = f"tag info, loop time: {count}"
            user_response = f"tag info: skill problem, skill {command.lower()}"
        else:
            user_response = f"Alexa, {command}"

        # Here we pass the appropriate arguments to write_to_file
        target_df = write_to_file(target_df, invoke_to_skill_id, invoke_to_skill_name, speech, sentence, out, user_response, row)
        target_df.to_csv(target, index=False)
        count += 1

    return target_df


def main():
    test_voice_folder = voice_folder
    test_results_folder = target_folder
    wav_files = get_wav_files(test_voice_folder)

    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
        
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



        openai.api_key = key
        api_key = key
        chatbot = Chatbot(api_key)

        # chatbot.set_context("This chatbot is designed to interact with Alexa. It should provide concise, relevant, and  adhere strictly to the text provided without extra input.")

        
        speaker_voice = wav
        print("=====================================")
        print("use %s voice to start testing..." % wav_file)
        print("=====================================")

        source_df, target_df, invoke_to_skill_id, invoke_to_skill_name = read_from_files(target)

        for i,row in source_df.iterrows():

            interact_with_skill(model, processor, device, tts, speaker_voice, chatbot, source_df, row)

            # Step 2: "Alexa, Stop" and record
            target_df = execute_command_and_record("Stop", model, processor, device, tts, speaker_voice, target_df, invoke_to_skill_id, invoke_to_skill_name, row)

            # Step 3: Reactivate and interact with the skill again
            interact_with_skill(model, processor, device, tts, speaker_voice, chatbot, source_df, row)

            # Step 4: "Alexa, Exit" and record
            target_df = execute_command_and_record("Exit", model, processor, device, tts, speaker_voice, target_df, invoke_to_skill_id, invoke_to_skill_name, row)

            # Step 5: Reactivate and interact with the skill again
            interact_with_skill(model, processor, device, tts, speaker_voice, chatbot, source_df, row)

            # Step 6: "Alexa, Cancel" and record
            target_df = execute_command_and_record("Cancel", model, processor, device, tts, speaker_voice, target_df, invoke_to_skill_id, invoke_to_skill_name, row)

        print("%s voice play complete!"%wav_file)
        # Start the sleep countdown
        # restart_countdown(40)


if __name__ == "__main__":
    main()
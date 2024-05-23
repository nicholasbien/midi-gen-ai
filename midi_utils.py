import time
from datetime import datetime, timedelta
import mido
from mido import Message
import requests

from apscheduler.schedulers.background import BackgroundScheduler

from transformers import GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from convert import expand_condensed_sequence

def send_note_on(note, outport):
    note_on = Message('note_on', note=note.pitch, velocity=note.velocity)
    print(f"Sending MIDI message: {note_on}")
    outport.send(note_on)

def send_note_off(note, outport):
    note_off = Message('note_off', note=note.pitch, velocity=note.velocity)
    print(f"Sending MIDI message: {note_off}")
    outport.send(note_off)

def send_notes(note_sequence, scheduler, outport):
    start_time = datetime.now()
    for note in note_sequence.notes:

        # Calculate the start time of the note
        note_start_time = start_time + timedelta(seconds=note.start_time)
        # Schedule the note_on message
        scheduler.add_job(send_note_on, 'date', run_date=note_start_time, args=[note, outport])
        
        # Calculate the end time of the note
        note_end_time = start_time + timedelta(seconds=note.end_time)
        # Schedule the note_off message
        scheduler.add_job(send_note_off, 'date', run_date=note_end_time, args=[note, outport])

def send_midi_from_file(filename, url):
    # Open a MIDI output port
    # print(mido.get_output_names())
    outport = mido.open_output("IAC Driver Bus 2")  # Use Bus 2 for output
    scheduler = BackgroundScheduler()
    scheduler.start()

    with open(filename, 'r') as file:
        condensed_sequence = file.read()

    # Play midi events from .txt
    #note_sequence = expand_condensed_sequence(condensed_sequence)

    # Generate based on .txt and play generation
    query_txt = condensed_sequence
    # response_txt, decoded_query_tensor = generate(tokenizer, model, query_txt, ppo_trainer)
    # combine the query and response txt to create the full note sequence
    # combined_txt = query_txt + response_txt
    # call generate API

    with open(filename, 'r') as file:
        condensed_sequence = file.read()

    # Prepare the data to send to the external API
    data = {
        'text': condensed_sequence  # Assuming the API expects the text to generate from under the key 'text'
    }

    # Make a POST request to the external API
    response = requests.post(url, json=data, verify=False)

    if response.status_code == 200:
        # Assuming the API returns a JSON response with the generated text under the key 'generated_text'
        response_data = response.json()
        response_txt = response_data.get('generated_text', '')
        print(response_txt)
        # Use the generated text to create the full note sequence
        note_sequence = expand_condensed_sequence(response_txt)
        send_notes(note_sequence, scheduler, outport)
    else:
        print(f"Failed to call generate API: {response.status_code} - {response.text}")

    # Keep the script running until all jobs are done
    while scheduler.get_jobs():
        time.sleep(1)  # Sleep for a short period to prevent high CPU usage

    scheduler.shutdown()  # Shutdown the scheduler after all jobs are done

if __name__=="__main__":
    send_midi_from_file("txt/a_organ.txt")

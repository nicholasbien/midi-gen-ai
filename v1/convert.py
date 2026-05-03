import argparse
import os
from note_seq import midi_io
from note_seq.protobuf import music_pb2

def condense_note_sequence(note_sequence, time_divisor=1.0, velocity_divisor=1.0):
    """
    Convert a NoteSequence into a condensed text format.

    Parameters:
    - note_sequence: The NoteSequence object to condense.
    - time_divisor: Divisor to scale time values, making them more compact.
    - velocity_divisor: Divisor to scale velocity values, making them more compact.

    Returns:
    A string representing the condensed NoteSequence.
    """
    condensed_notes = []

    for note in note_sequence.notes:
        # Scale and round time and velocity for more compact representation
        start_time = round(note.start_time / time_divisor, 2)
        duration = round((note.end_time - note.start_time) / time_divisor, 2)
        velocity = round(note.velocity / velocity_divisor, 2)

        # Create a condensed representation of the note
        condensed_note = f"{note.pitch}_{start_time}_{duration}_{velocity}"
        condensed_notes.append(condensed_note)

    # Join all condensed notes with a space (or another delimiter of your choice)
    condensed_sequence = ' '.join(condensed_notes)
    return condensed_sequence

def expand_condensed_sequence(condensed_sequence, time_multiplier=1.0, velocity_multiplier=1.0):
    """
    Convert a condensed sequence string back to a NoteSequence object.

    Parameters:
    - condensed_sequence: The condensed string representation of the NoteSequence.
    - time_multiplier: Multiplier to scale time values back to their original scale.
    - velocity_multiplier: Multiplier to scale velocity values back to their original scale.

    Returns:
    A NoteSequence object reconstructed from the condensed sequence.
    """
    note_sequence = music_pb2.NoteSequence()

    # Split the condensed sequence into individual notes
    condensed_notes = condensed_sequence.split(' ')

    for note_str in condensed_notes:
        # Split each note into its components
        try:
            pitch, start_time, duration, velocity = note_str.split('_')

            # Convert string values back to integers and scale time and velocity
            pitch = int(pitch)
            start_time = float(start_time) * time_multiplier
            end_time = start_time + (float(duration) * time_multiplier)
            velocity = int(float(velocity) * velocity_multiplier)

            # Add the note to the NoteSequence
            note = note_sequence.notes.add()
            note.pitch = pitch
            note.start_time = start_time
            note.end_time = end_time
            note.velocity = velocity

        except:
            ### TODO: better handling for empty and truncated notes
            continue
    # Set total time of the sequence
    note_sequence.total_time = max(note.end_time for note in note_sequence.notes)

    return note_sequence

def process_files_recursive(mode, input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        # Sort directories
        dirs.sort()

        # Sort files
        files.sort()

        # Determine the current output directory
        relative_path = os.path.relpath(root, input_folder)
        current_output_folder = os.path.join(output_folder, relative_path)

        # Create the output directory if it doesn't exist
        os.makedirs(current_output_folder, exist_ok=True)

        if mode == 'encode':
            for file in files:
                if file.endswith('.mid'):
                    process_file_encode(os.path.join(root, file), current_output_folder)
        elif mode == 'decode':
            for file in files:
                if file.endswith('.txt'):
                    process_file_decode(os.path.join(root, file), current_output_folder)

def process_file_encode(input_path, output_folder):
    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0] + '.txt')
    print(f"Encoding {input_path} to {output_path}")
    try:
        note_sequence = midi_io.midi_file_to_note_sequence(input_path)
        condensed_sequence = condense_note_sequence(note_sequence)
        with open(output_path, 'w') as file:
            file.write(condensed_sequence)
    except Exception as e:
        print(e)
    return output_path

def process_file_decode(input_path, output_folder, instrument=None):
    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0] + '.mid')
    with open(input_path, 'r') as file:
        condensed_sequence = file.read()
    note_sequence = expand_condensed_sequence(condensed_sequence)
    
    if instrument != None:

        # set the instrument type of each note

        is_drums = False

        # 1 Acoustic Grand Piano
        # 5 Electric Piano 1
        # 25 Acoustic Guitar
        # 34 Electric Bass
        program = 1  # default to piano
        if instrument == 'piano':
            program = 1
        elif instrument == 'electric piano':
            program = 5
        elif instrument == 'bass':
            program = 34
        elif instrument == 'drums':
            is_drums = True

        # A program selects an instrument's sound.
        # Note that the General MIDI documentation is 1-based, but this field is
        # 0-based. So GM documents program 12 as vibraphone, but this field would
        # be set to 11 for that instrument.
        # See www.midi.org/specifications/item/gm-level-1-sound-set.

        # convert to note_seq program
        program -= 1

        # Change instrument for all notes
        for note in note_sequence.notes:
            if is_drums:
                note.is_drum = True
            else:
                note.program = program
    
    midi_io.note_sequence_to_midi_file(note_sequence, output_path)

    print(f"Decoded {input_path} to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Encode MIDI files to a condensed format or decode them back, processing directories recursively.")
    parser.add_argument("mode", choices=["encode", "decode"], help="Operation mode: encode or decode.")
    parser.add_argument("input_folder", help="Input folder containing MIDI or text files.")
    parser.add_argument("output_folder", help="Output folder to save the processed files.")
    args = parser.parse_args()

    process_files_recursive(args.mode, args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()


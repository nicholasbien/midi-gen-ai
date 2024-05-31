import os
import glob
from xml.etree import ElementTree as ET
from xml.dom import minidom
import gzip
import base64

from convert import process_file_decode

# Function to extract MIDI data from .als file
def extract_midi_from_als(als_file_path, txt_output_dir, midi_output_dir):
    print(txt_output_dir, midi_output_dir)
    # Ensure the output directory exists
    os.makedirs(txt_output_dir, exist_ok=True)
    os.makedirs(midi_output_dir, exist_ok=True)

    extracted_midis = []
    # Step 1: Unzip the .als file
    with gzip.open(als_file_path, 'rb') as gzipped_file:
        xml_content = gzipped_file.read()

    # Step 2: Parse the XML content
    root = ET.fromstring(xml_content)
    # ET.dump(root)

    def prettify(element):
        rough_string = ET.tostring(element, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    # Function to recursively print element names
    def print_element_names(element, level=0):
        # Print the current element's tag with indentation based on its level in the tree
        #if 'Midi' in element.tag or 'Track' in element.tag:
        #    print('  ' * level + element.tag)
        # if 'Tempo' in element.tag:
        
        # print the attributes and values
        #if True:
        #    attribs_print = ' '.join([f'{k}: {element.attrib[k]}' for k in element.attrib.keys()]) if element.attrib else ''
        #    print(' ' * level + element.tag + ': ' + attribs_print)
        
        # Recursively call this function for each child element
        for child in element:
            print_element_names(child, level + 1)

    # Start printing from the root element
    print_element_names(root)
    # Iterate through each KeyTrack in the XML
    midi_file_count = 0
    txt_strs = set()
    tempo = root.find('.//Tempo')
    tempo_manual = tempo.find('.//Manual')
    tempo_value = float(tempo_manual.attrib['Value'])
    print(f'Tempo: {tempo_value}')
    for midi_clip in root.findall('.//MidiClip'):
        # print(midi_clip)
        # print(midi_file_count)
        note_sequence = []
        for key_track in midi_clip.findall('.//KeyTrack'):
            pitch = int(key_track.find('MidiKey').attrib['Value'])
            notes = key_track.find('Notes').findall('MidiNoteEvent')
            for note in notes:
                start_time = round(float(note.attrib['Time']), 2)
                duration = round(float(note.attrib['Duration']), 2)
                # Ableton represents time in terms of beats
                # so it's as if the song is at 60bpm
                # adjust duration and start_time to account for tempo difference
                # the ideal solution here is to also encode the tempo into the song
                # also there might be intricacies with the tempo the sample was
                # recorded at vs. the master tempo of the song
                start_time = (60.0/tempo_value) * start_time
                duration = (60.0/tempo_value) * duration
                # TODO: fix how velocity is encoded from Midi
                #velocity = int(round(float(note.attrib['Velocity']), 0))
                velocity = round(float(note.attrib['Velocity']), 2)
                note_sequence.append((pitch, start_time, duration, velocity))
        # skip empty midi clips
        if len(note_sequence) == 0:
            # print("Empty Midi Clip")
            continue
        # sort by start_time
        note_seq = sorted(note_sequence, key=lambda x: x[1])
        condensed_notes = []
        for note_tuple in note_sequence:
            pitch, start_time, duration, velocity = note_tuple
            condensed_note = f"{pitch}_{start_time}_{duration}_{velocity}"
            condensed_notes.append(condensed_note)
        condensed_sequence = ' '.join(condensed_notes)
        # print(condensed_sequence)
        if condensed_sequence not in txt_strs:
            txt_strs.add(condensed_sequence)
            project_name_in_file = os.path.basename(txt_output_dir).replace(' ', '_')
            txt_output_path = os.path.join(txt_output_dir, f"{project_name_in_file}_{midi_file_count}.txt")
            with open(txt_output_path, 'w') as file:
                file.write(condensed_sequence)
            process_file_decode(txt_output_path, midi_output_dir)
        midi_file_count += 1
    return extracted_midis

def process_ableton_projects(project_folders, txt_output_dir, midi_output_dir):
    # Ensure the output directory exists
    os.makedirs(txt_output_dir, exist_ok=True)
    os.makedirs(midi_output_dir, exist_ok=True)
    for project_folder in project_folders:
        project_name = os.path.basename(project_folder)
        txt_project_output_dir = os.path.join(txt_output_dir, project_name)
        midi_project_output_dir = os.path.join(midi_output_dir, project_name)
        os.makedirs(txt_project_output_dir, exist_ok=True)
        os.makedirs(midi_project_output_dir, exist_ok=True)

        # Find all .als files in the current project folder
        print(project_folder)
        print(midi_project_output_dir)
        als_files = glob.glob(os.path.join(project_folder, "*.als"))
        # print(als_files)
        for als_file in als_files:
            # Create a unique output directory for each .als file based on its name
            txt_set_output_dir = os.path.join(txt_project_output_dir, os.path.splitext(os.path.basename(als_file))[0])
            midi_set_output_dir = os.path.join(midi_project_output_dir, os.path.splitext(os.path.basename(als_file))[0])
            print(txt_output_dir, midi_output_dir)
            extract_midi_from_als(als_file, txt_set_output_dir, midi_set_output_dir)

# Example usage
project_folders = [
    '/Users/nicholasbien/Music/Ableton/wagner Project',
    '/Users/nicholasbien/Music/Ableton/process Project',
    # Add more project folders as needed
]
process_ableton_projects(project_folders, txt_output_dir="my_txt", midi_output_dir="my_midi")


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load an audio file (replace 'audio_file.wav' with your file)
audio_path = './data/3_Seven Nation Army_The White Stripes.wav'
y, sr = librosa.load(audio_path, sr=None)

# Step 1: Harmonic-percussive separation
y_harmonic, _ = librosa.effects.hpss(y)

# Step 2: Compute chroma features
chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# Save chroma features to a file
with open('chroma_features.txt', 'w') as f:
    f.write("Chroma Features Analysis\n")
    f.write("=======================\n\n")
    f.write("Time-based chroma intensities:\n")
    #f.write("Time (s)\t" + "\t".join(chords) + "\n")
    
    # Get time points
    frames = range(chroma.shape[1])
    times = librosa.frames_to_time(frames, sr=sr)
    f.write("T\tC\tC#\tD\tD#\tE\tF\tF#\tG\tG#\tA\tA#\tB\n")    
    # Write data for each time frame
    for t, frame in zip(times, chroma.T):
        if t % 1 < 0.1:  # Write data roughly every second
            f.write(f"{t:.2f}\t")
            f.write("\t".join([f"{val:.3f}" for val in frame]))
            dominant_idx = np.argmax(frame)
            f.write(f"\t{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][dominant_idx]}\n")




    #f.write("Time (s)\t" + "\t".join(chords) + "\n")
    with open('chroma_result.txt', 'w') as f:
        f.write("Chroma Features Analysis\n")
        f.write("=======================\n\n")
        f.write("Time-based chroma intensities:\n")   
        # Get time points
        frames = range(chroma.shape[1])
        times = librosa.frames_to_time(frames, sr=sr)
        # Write data for each time frame
        for t, frame in zip(times, chroma.T):
            if t % 1 < 0.1:  # Write data roughly every second
                f.write(f"{t:.2f}\t")
                dominant_idx = np.argmax(frame)
                f.write(f"\t{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][dominant_idx]}\n")

print("Chroma features saved to chroma_features.txt")


#print to a json file as well
import json

# Create a list to store the data
json_data = []

# Get time points for JSON
frames = range(chroma.shape[1])
times = librosa.frames_to_time(frames, sr=sr)

# Collect data for each time frame
for t, frame in zip(times, chroma.T):
    if t % 1 < 0.1:  # Store data roughly every second
        dominant_idx = np.argmax(frame)
        frame_data = {
            "time": round(float(t), 2),
            "intensities": {
                "C": float(frame[0]),
                "C#": float(frame[1]), 
                "D": float(frame[2]),
                "D#": float(frame[3]),
                "E": float(frame[4]),
                "F": float(frame[5]),
                "F#": float(frame[6]),
                "G": float(frame[7]),
                "G#": float(frame[8]),
                "A": float(frame[9]),
                "A#": float(frame[10]),
                "B": float(frame[11])
            },
            "dominant_note": ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][dominant_idx]
        }
        json_data.append(frame_data)

# Write to JSON file
with open('chroma_features.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=4)





# Step 3: Plot the chroma features
plt.figure(figsize=(10, 6))
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr, cmap='coolwarm')
plt.colorbar()
plt.title('Chroma Features')
plt.xlabel('Time (s)')
plt.ylabel('Chroma')
plt.show()

# Step 4: Simple chord estimation
# Use the mean chroma vector to estimate the most prominent pitch class
mean_chroma = chroma.mean(axis=1)
chords = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Identify the most prominent note
prominent_chord_idx = np.argmax(mean_chroma)
print(f"The most prominent chord is: {chords[prominent_chord_idx]}")


# Extract times and dominant notes from json_data
times = [frame['time'] for frame in json_data]
dominant_notes = [frame['dominant_note'] for frame in json_data]

# Convert notes to numeric values for plotting
note_to_num = {note: i for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}
dominant_notes_num = [note_to_num[note] for note in dominant_notes]


# Define colors for each note
note_colors = {
    'C': '#FF0000',    # Red
    'C#': '#FF7F00',   # Orange
    'D': '#FFFF00',    # Yellow
    'D#': '#7FFF00',   # Chartreuse
    'E': '#00FF00',    # Green
    'F': '#00FF7F',    # Spring Green
    'F#': '#00FFFF',   # Cyan
    'G': '#007FFF',    # Azure
    'G#': '#0000FF',   # Blue
    'A': '#7F00FF',    # Violet
    'A#': '#FF00FF',   # Magenta
    'B': '#FF007F'     # Rose
}

# Create a list of colors matching the dominant notes
plot_colors = [note_colors[note] for note in dominant_notes]

# Plot each segment with its corresponding color
for i in range(len(times)-1):
    plt.plot(times[i:i+2], dominant_notes_num[i:i+2], 
             color=plot_colors[i], 
             linewidth=2)

# Plot dominant notes over time 
plt.figure(figsize=(10, 6))

# Extract times and dominant notes from json_data
times = [frame['time'] for frame in json_data]
dominant_notes = [frame['dominant_note'] for frame in json_data]

# Convert notes to numeric values for plotting
note_to_num = {note: i for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}
dominant_notes_num = [note_to_num[note] for note in dominant_notes]

# Create the plot
plt.plot(times, dominant_notes_num, 'b.-', markersize=8)
plt.yticks(range(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.title('Dominant Notes Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Note')
plt.show()

import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load an audio file (replace 'audio_file.wav' with your file)
audio_path = './data/3_Seven Nation Army_The White Stripes.wav'
y, sr = librosa.load(audio_path, sr=None)

# Calculate RMS energy
rms = librosa.feature.rms(y=y)

# Plot the RMS energy over time
frames = range(len(rms[0]))
times = librosa.frames_to_time(frames, sr=sr)
print("frames",frames)
print("times",times)
print("rms",rms[0])

# Save numeric results to file
with open('rms_results.txt', 'w') as f:
    f.write("RMS Energy Analysis Results\n")
    f.write("==========================\n\n")
    f.write(f"Number of frames: {len(frames)}\n")
    f.write(f"Total duration: {times[-1]:.2f} seconds\n")
    f.write(f"Maximum RMS energy: {np.max(rms[0]):.4f}\n")
    f.write(f"Minimum RMS energy: {np.min(rms[0]):.4f}\n")
    f.write(f"Average RMS energy: {np.mean(rms[0]):.4f}\n\n")
    interval = np.max(rms[0]) - np.min(rms[0]) 
    normalizationfactor = 6/ interval
    f.write("Frame-by-frame data:\n")
    f.write("Time (s)\tRMS Energy\n")
    for i, (t, r) in enumerate(zip(times, rms[0])):
        if i % 100 == 0:
            s=round(r*normalizationfactor,0)
            if s>5 :
                s=5 
            f.write(f"{t:.3f}\t{r:.6f}'      '\t{s:.00f}\n")

# Save numeric results to XML file
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Create the root element
root = ET.Element("RMS_Analysis")

# Add basic statistics
stats = ET.SubElement(root, "Statistics")
ET.SubElement(stats, "Frames").text = str(len(frames))
ET.SubElement(stats, "Duration").text = f"{times[-1]:.2f}"
ET.SubElement(stats, "MaxRMS").text = f"{np.max(rms[0]):.4f}"
ET.SubElement(stats, "MinRMS").text = f"{np.min(rms[0]):.4f}"
ET.SubElement(stats, "AverageRMS").text = f"{np.mean(rms[0]):.4f}"

# Add frame data
frames_data = ET.SubElement(root, "FrameData")
interval = np.max(rms[0]) - np.min(rms[0])
normalizationfactor = 6 / interval

for i, (t, r) in enumerate(zip(times, rms[0])):
    if i % 100 == 0:
        s = round(r*normalizationfactor, 0)
        if s > 5:
            s = 5
        frame = ET.SubElement(frames_data, "Frame")
        ET.SubElement(frame, "Time").text = f"{t:.3f}"
        ET.SubElement(frame, "RMSEnergy").text = f"{r:.6f}"
        ET.SubElement(frame, "NormalizedScore").text = f"{s:.0f}"

# Create a pretty-printed XML string
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")

# Save to file
with open('rms_results.xml', 'w') as f:
    f.write(xmlstr)


# Save to XML file
with open('rms_results.xml', 'w') as f:
    f.write(xmlstr)

# Create and save JSON file
import json

json_data = {
    "statistics": {
        "frames": len(frames),
        "duration": f"{times[-1]:.2f}",
        "max_rms": f"{np.max(rms[0]):.4f}",
        "min_rms": f"{np.min(rms[0]):.4f}",
        "average_rms": f"{np.mean(rms[0]):.4f}"
    },
    "frame_data": []
}

# Add frame data
interval = np.max(rms[0]) - np.min(rms[0])
normalizationfactor = 6 / interval

for i, (t, r) in enumerate(zip(times, rms[0])):
    if i % 100 == 0:
        s = round(r*normalizationfactor, 0)
        if s > 5:
            s = 5
        json_data["frame_data"].append({
            "time": f"{t:.3f}",
            "rms_energy": f"{r:.6f}",
            "normalized_score": int(s)
        })

# Save to JSON file
with open('rms_features.json', 'w') as f:
    json.dump(json_data, f, indent=4)

# // (plt.figure etc.) ...


plt.figure(figsize=(10, 6))
plt.plot(times, rms[0], label="RMS Energy")
plt.title("RMS Energy Over Time")
plt.xlabel("Time (s)")
plt.ylabel("RMS Energy")
plt.legend()
plt.show()

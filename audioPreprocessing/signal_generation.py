import matplotlib.pyplot as plt
import numpy as np
import wave

obj = wave.open("../data/train/A1.wav", 'rb')

# Extract parameters
sample_freq = obj.getframerate()
n_samples = obj.getnframes()
sample_width = obj.getsampwidth()
signal_wave = obj.readframes(n_samples)
obj.close()

# Convert signal to NumPy array
signal_array = np.frombuffer(signal_wave, dtype=np.int16)

# Normalize the signal to range [-1, 1]
signal_array = signal_array / np.max(np.abs(signal_array))

# Generate time axis
t_audio = n_samples / sample_freq
time = np.linspace(0, t_audio, num=n_samples)

# Plot the normalized audio signal
plt.figure(figsize=(9, 4))
plt.plot(time, signal_array)
plt.title("Normalized Audio Signal")
plt.xlabel("Time [s]")
plt.ylabel("Normalized Amplitude")
plt.xlim(0, t_audio)
plt.ylim(-1, 1)

# Display only -1, 0, and 1 on y-axis
plt.yticks([-1, 0, 1])

plt.show()

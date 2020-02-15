import os
import shutil
import tkinter
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


def split_by_class_name():
    if not os.path.isdir("data"):
        os.mkdir("data")
    count = 0
    for audio_file in os.listdir("audio"):
        # Getting the class name of the audio file
        file_name = audio_file.strip(".wav")
        dir = file_name.split("-")[-1]

        sample_rate, samples = wavfile.read('audio/' + audio_file)
        plt.specgram(samples, Fs=sample_rate, NFFT=512)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.xlim(left=0, right=5)
        # plt.show()
        if not os.path.isdir("data/" + dir):
            os.mkdir("data/" + dir)
        plt.savefig("data/" + dir + "/" + file_name)
        count += 1
        print(count)


if __name__ == "__main__":
    split_by_class_name()

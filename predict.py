from tensorflow import keras
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import PIL.Image
import numpy as np

class Prediction:
    def __init__(self):
        self.model = keras.models.load_model('cnn_spectrogram_to_digit.h5')

    def convert_to_spectrogram(self, sours_dir, to_safe_dir, spectrogram_dimensions=(34, 50)):
        try:
            sample_rate, samples = wav.read(sours_dir)       
            print(samples)      
            fig = plt.figure()
            fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.specgram(samples, cmap='gray_r', Fs=2, noverlap=16)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            fig.savefig('test.png', bbox_inches="tight", pad_inches=0)
            plt.close()
        except UnboundLocalError:
            print("Problem with " , sours_dir ,  " file")

    def predictNumber(self):
        self.convert_to_spectrogram('audio.wav', '')

        rgba_image = PIL.Image.open('test.png')
        rgb_image = rgba_image.convert('RGB')
        data = np.asarray(rgb_image)
        data = np.reshape(data, (34, 50 ,3))
        img = (np.expand_dims(data,0))
        predictions = self.model.predict(img)
        print(np.argmax(predictions))
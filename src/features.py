import librosa as lb
import numpy as np
from pathlib import Path
import joblib
from sklearn.neighbors import KNeighborsClassifier

class Features:

    def __init__(self, num_mfcc, duration, audio_data, sample_rate):

        self.NUM_MFCC = num_mfcc
        self.duration = duration

        self.y = audio_data
        self.sr = sample_rate

    def extract_features(self):
        """
        Extracts MFCC features Loads .wav audio files from a specified location Zero-pads the feature vectors to equalize their lengths and transforms the MFCC matrix
        into a 1D vector of size NUM_MFCC x number_of_frames.
        """
        #y, sr = self.load_audio_with_padding()
        y = self.y
        sr = self.sr

        print("extracting features", sr)
        features = lb.feature.mfcc(y=y, sr=sr, n_mfcc=self.NUM_MFCC, n_fft=2048, hop_length=512)

        print("size features", features.size)
        return features

    def load_audio_with_padding(self):
        """
        Loads .wav audio files from a specified location Zero-pads the feature vectors to
        equalize their lengths and transforms the MFCC matrix
        into a 1D vector of size NUM_MFCC x number_of_frames.
        """

        # Get actual duration of the loaded audio
        actual_duration = lb.get_duration(y=self.y, sr=self.sr)

        # Check if the actual duration is less than the target duration
        if actual_duration < self.duration:
            # Calculate the number of samples needed for padding
            samples_to_pad = int((self.duration - actual_duration) * self.sr)

            # Create zero signal to append
            zero_signal = np.zeros(samples_to_pad)

            # Append zero signal to the audio
            y_padded = np.concatenate([self.y, zero_signal])

            return y_padded, self.sr

        else:
            target_samples = int(self.duration * self.sr)
            y = self.y[:target_samples]  # Tr
            return y, self.sr

    def collect_features(self):

        try:
            features = self.extract_features()
            print("success extraction")

        except Exception as e:
            print(str(e))

        sample_len = 469 # magic number, comes from model training
        features_padded = (self.zero_padding(features, self.NUM_MFCC, sample_len))

        return features_padded

    @staticmethod
    def predict(features_padded, knn_model, scaler):

        new_sample = features_padded

        scaled_sample = scaler.transform(new_sample.reshape(1, -1))

        predictions = knn_model.predict(scaled_sample)
        return predictions

    @staticmethod
    def zero_padding(features, NUM_MFCC, sample_len):

        max_size = sample_len
        features_padded = []
        print("features", features.shape)
        if (max_size - features.shape[1] <= 0):

            for x in features:
                features_padded.append(x[0:max_size])

        else:
            features_padded = features
            print("test", NUM_MFCC, max_size , features.shape[1])
            diff = max_size - features.shape[1]
            mfccs_3 = np.zeros(( NUM_MFCC, diff ))
            features_padded = np.hstack((features, mfccs_3))

        reshaped_array = (np.array(features_padded)).flatten()
        print("size feature_padded", reshaped_array.size )

        return reshaped_array

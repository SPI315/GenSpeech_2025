#!/usr/bin/env python3
import torch
import torchaudio
from TTS.api import TTS
from TTS.tts.utils.synthesis import synthesis


class TextToSpecConverter:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/fast_pitch"):
        self.model_name = model_name
        self.tts_handler = TTS(model_name=model_name)
        print('**' * 25,
              'INFO: discard default vocoder initialization messages above',
              '**' * 25, sep='\n')
        self.model = self.tts_handler.synthesizer.tts_model
        self.config = self.tts_handler.synthesizer.tts_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_cuda = True if self.device == "cuda" else False
        print(f"Model {model_name} loaded")
    
    def text2spec(self, text: str):
        """
        Convert text to mel spectrogram using pretrained TTS model.
        Args:
            text (str): Input text to convert to mel spectrogram
        Returns:
            mel_spec (numpy.ndarray): Mel spectrogram of the input text
                with shape [C, T] = [num_mel_channels, num_frames]
        """
        outputs = synthesis(
            self.model,
            text,
            self.config,
            self.use_cuda,
            use_griffin_lim=False,
            do_trim_silence=False
        )
        mel_spec = outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()
        # denormalize tts output based on the tts audio config
        mel_spec = self.model.ap.denormalize(mel_spec.T).T
        return mel_spec


def save_spectrogram(mel_spec, filename="spectrogram.png", title="Mel Spectrogram"):
    """
    Save the mel spectrogram as an image file
    
    Args:
        mel_spec: The mel spectrogram to save
        filename: Output filename
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Frames')
    plt.ylabel('Mel Channels')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Spectrogram saved to {filename}")


def melspec_to_audio_default_vocoder(t2s, mel_spec, filename='output.wav'):
    vocoder_input = t2s.tts_handler.synthesizer.vocoder_ap.normalize(mel_spec.T)
    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
    # [1, C, T]
    waveform = t2s.tts_handler.synthesizer.vocoder_model.inference(vocoder_input.to("cpu"))
    waveform = waveform.squeeze(0)
    torchaudio.save(filename, waveform.cpu(), 22050)
    print(f"Audio saved to {filename}")


if __name__ == "__main__":
    # Example usage
    t2s = TextToSpecConverter()
    mel_spec = t2s.text2spec("Hello, world!")
    
    save_spectrogram(mel_spec, filename="output_spectrogram.png", title="Mel Spectrogram of 'Hello, world!'")

    # If you want to validate mel_spec with default vocoder, use the code sample below
    melspec_to_audio_default_vocoder(t2s, mel_spec, filename='output.wav')

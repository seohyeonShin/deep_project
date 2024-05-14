
import torch
import torch.nn as nn
import torchaudio as ta
import json

# For Grad-TTS
from .model import GradTTS
from .text import text_to_sequence
from .text.korean_dict import ALL_SYMBOLS as symbols

# For HiFi-GAN
import sys
sys.path.append('./hifi-gan/')
from .env import AttrDict
from .models import Generator as HiFiGAN
        

class PerturbationGradTTS():
    def __init__(self, model_path, hifi_config_path, hifi_path, spk_emb_dim=192, n_enc_channels=192, filter_channels=768,
                 filter_channels_dp=256, n_heads=2, n_enc_layers=6,enc_kernel=3, enc_dropout=0.1, window_size=4, 
                 n_feats=80, dec_dim=256, beta_min=0.05, beta_max=20.0, pe_scale=1000):
        
        self.device = self.set_device()
        self.generator = GradTTS(len(symbols)+1, spk_emb_dim, n_enc_channels, filter_channels,
                                 filter_channels_dp, n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                                 n_feats, dec_dim, beta_min, beta_max, pe_scale=pe_scale)
        self.generator.load_state_dict(torch.load(model_path, map_location=lambda loc, storage: loc))
        self.generator = self.generator.to(self.device).eval()

        self.hifi = self.set_hifigan(hifi_config_path, hifi_path)
        
        
    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Device is cuda')
        else:
            device = torch.device('cpu')
            print('Device is cpu')

        return device
        
    def set_hifigan(self, config_path, checkpts_path):
        with open(config_path) as f:
            h = AttrDict(json.load(f))
        hifigan = HiFiGAN(h)
        hifigan.load_state_dict(torch.load(checkpts_path, map_location=lambda loc, storage: loc)['generator'])
        _ = hifigan.cuda().eval()
        hifigan.remove_weight_norm()
        
        return hifigan

    def get_text(self, text):
        x = torch.LongTensor(self.intersperse(text_to_sequence(text, ['korean_cleaners']), len(symbols))).cuda()[None]
        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

        return x, x_lengths
    
    
    def intersperse(self, lst, item):
        # Adds blank symbol
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result                     
    
    # text와 audio를 입력받아 wav를 생성하는 함수
    def generate_speech(self, text, n_timesteps=10, temperature=1.3, length_scale=0.91):
        # Function that generates only one sample
        x, x_lengths = self.get_text(text)
        
        y_enc, y_dec, attn = self.generator.forward_pass(x.to(self.device), x_lengths.to(self.device), n_timesteps=n_timesteps, temperature=temperature,
                                       stoc=False, spk=None,
                                       length_scale=length_scale)
        with torch.no_grad():
            audio = self.hifi.forward(y_dec).cpu().squeeze().clamp(-1, 1)
        
        print('Done!')
        return audio
    
    def test_sample(self):
        text = 'It was like the reflections from a score of mirrors placed round the walls at different angles.'
        generated_wav = self.generate_speech(text, n_timesteps=100)
        return generated_wav
        
if __name__ == '__main__':
    model = PerturbationGradTTS(model_path='./checkpts/grad_140(500h)_koeran_tp.pt', hifi_config_path='./checkpts/config.json', hifi_path='./checkpts/g_02500000')
    generated_wav = model.test_sample()
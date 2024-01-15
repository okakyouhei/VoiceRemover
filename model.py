import playsound as ps
from pydub import AudioSegment
from moviepy.editor import *
import os
from moviepy.editor import AudioFileClip, VideoFileClip
import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from lib import dataset
from lib import nets
from lib import spec_utils
from lib import utils

from inference import Separator

#mp4 changes to mp3 file
def mp4ToMp3(mp4file, mp3file): 
    videoclip = VideoFileClip(mp4file)
    audioclip = videoclip.audio
    audioclip.write_audiofile("input.mp3") #mp3file を "input.mp3"に変更した
    audioclip.close()
    videoclip.close()

#Flask（app.py）には書かない
#music = "input.mp3"
#video = input("Enter the output video path: ")
    
def voiceRemove(video, music):
    #(mp4 filename, mp3 filename that made by it)
    mp4ToMp3(video, music)

    #extract length of completed mp3file
    sound = AudioSegment.from_file(music,"mp3")
    #print the length
    time = sound.duration_seconds
    print("-------------------------------")
    print('playTime: ' + str(time))
    print("-------------------------------")

    #ps.playsound("input.mp3")

    #content of inference.py
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=0)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    p.add_argument('--input', '-i', required=False)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--output_image', '-I', action='store_true', default=False)
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--output_dir', '-o', type=str, default="output")
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedNet(args.n_fft, 32, 128)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.gpu))
            model.to(device)
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
    basename = os.path.splitext(os.path.basename(args.input))[0]
    print('done')

    if X.ndim == 1:
        # mono to stereo
        X = np.asarray([X, X])

    print('stft of wave source...', end=' ')
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    print('done')

    sp = Separator(model, device, args.batchsize, args.cropsize, args.postprocess)

    if args.tta:
        y_spec, v_spec = sp.separate_tta(X_spec)
    else:
        y_spec, v_spec = sp.separate(X_spec)

    print('validating output directory...', end=' ')
    output_dir = args.output_dir
    if output_dir != "":  # modifies output_dir if theres an arg specified
        output_dir = output_dir.rstrip('/') + '/'
        os.makedirs(output_dir, exist_ok=True)
    print('done')

    print('inverse stft of instruments...', end=' ')
    wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}{}_Instruments.wav'.format(output_dir, basename), wave.T, sr)

    print('inverse stft of vocals...', end=' ')
    wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}{}_Vocals.wav'.format(output_dir, basename), wave.T, sr)

    if args.output_image:
        image = spec_utils.spectrogram_to_image(y_spec)
        utils.imwrite('{}{}_Instruments.jpg'.format(output_dir, basename), image)

        image = spec_utils.spectrogram_to_image(v_spec)
        utils.imwrite('{}{}_Vocals.jpg'.format(output_dir, basename), image)


    #mp3 + mp4
    org_video_path = video
    audio_path = "output/input_Instruments.wav"
    final_video_path = './static/products' #input("Enter the output folder path: ")
    final_video_name = "completedVideo.mp4" #出来た動画の名前
    start_dur = float(time - time)
    end_dur = float(time)

    final_video_path = os.path.join(final_video_path, final_video_name)

    video_clip = VideoFileClip(org_video_path)

    background_audio_clip = AudioFileClip(audio_path)
    bg_music = background_audio_clip.subclip(start_dur, end_dur)

    final_clip = video_clip.set_audio(bg_music)
    final_clip.write_videofile(final_video_path, codec='libx264', audio_codec="aac")
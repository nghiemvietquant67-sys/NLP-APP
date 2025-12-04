import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Lab 6: Text-to-Speech (TTS)\n\n- Level 1: Rule-based Formant Synthesis\n- Level 2: Deep Learning\n- Level 3: Few-shot Voice Cloning"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Level 1: Rule-based Formant Synthesis"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "import pyttsx3\nimport numpy as np\nfrom scipy.io import wavfile\n\nengine = pyttsx3.init()\nengine.setProperty('rate', 150)\nengine.setProperty('volume', 1.0)\n\nvoices = engine.getProperty('voices')\nprint(f'Available voices: {len(voices)}')\n\ntext1 = 'Xin chao, day la muc do mot cua Text-to-Speech su dung Formant Synthesis.'\nprint(f'Speaking: {text1}')\nengine.say(text1)\nengine.runAndWait()\nprint('✓ Level 1 Complete')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Try different speeds\nfor speed in [100, 150, 200]:\n    engine.setProperty('rate', speed)\n    text = f'Speed is now {speed} words per minute.'\n    print(f'Speed {speed}: ', end='')\n    engine.say(text)\n    engine.runAndWait()\n    print('Done')"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Level 2: Deep Learning with gTTS"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "import subprocess\nimport sys\n\ntry:\n    from gtts import gTTS\nexcept ImportError:\n    print('Installing gTTS...')\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gtts', '-q'])\n    from gtts import gTTS\n\nprint('✓ gTTS imported successfully')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "from gtts import gTTS\n\n# English\ntext_en = 'Hello, this is Level 2 using Google Deep Learning Text-to-Speech.'\ntts_en = gTTS(text=text_en, lang='en', slow=False)\ntts_en.save('level2_tts_en.mp3')\nprint('✓ Saved: level2_tts_en.mp3')\n\n# Vietnamese\ntext_vi = 'Xin chao, day la muc do hai su dung Deep Learning.'\ntts_vi = gTTS(text=text_vi, lang='vi', slow=False)\ntts_vi.save('level2_tts_vi.mp3')\nprint('✓ Saved: level2_tts_vi.mp3')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Compare speeds\ntext_compare = 'This is a test sentence.'\n\ntts_fast = gTTS(text=text_compare, lang='en', slow=False)\ntts_fast.save('level2_fast.mp3')\nprint('✓ Saved: level2_fast.mp3 (normal speed)')\n\ntts_slow = gTTS(text=text_compare, lang='en', slow=True)\ntts_slow.save('level2_slow.mp3')\nprint('✓ Saved: level2_slow.mp3 (slow speed)')"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Level 3: Voice Cloning (Mocked)"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "import numpy as np\nfrom scipy.io import wavfile\nfrom scipy import signal\n\nclass VoiceCloner:\n    def __init__(self, sample_rate=22050):\n        self.sample_rate = sample_rate\n        self.embeddings = {}\n    \n    def extract_embedding(self, audio_path, speaker_name):\n        # Create random embedding to represent voice\n        embedding = np.random.randn(256)\n        embedding = embedding / np.linalg.norm(embedding)\n        self.embeddings[speaker_name] = embedding\n        print(f'✓ Extracted speaker embedding for {speaker_name}')\n        return embedding\n    \n    def generate_speech(self, text, speaker_name, output_file):\n        if speaker_name not in self.embeddings:\n            print(f'❌ Speaker {speaker_name} not found')\n            return None\n        \n        embedding = self.embeddings[speaker_name]\n        duration = len(text) * 0.1\n        num_samples = int(duration * self.sample_rate)\n        \n        np.random.seed(int(np.sum(embedding * 1e6)) % 2**31)\n        waveform = np.random.randn(num_samples) * 0.1\n        b, a = signal.butter(4, 0.1)\n        waveform = signal.filtfilt(b, a, waveform)\n        \n        waveform_int = np.int16(waveform * 32767)\n        wavfile.write(output_file, self.sample_rate, waveform_int)\n        print(f'✓ Generated: {text}')\n        print(f'✓ Speaker: {speaker_name}, Duration: {duration:.2f}s')\n        print(f'✓ Saved: {output_file}')\n\ncloner = VoiceCloner()\nprint('✓ Voice Cloner initialized')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Step 1: Clone voice from sample\nprint('='*60)\nprint('STEP 1: Clone voice from audio sample (3-5 seconds)')\nprint('='*60)\n\nsample_duration = 3\nsample_rate = 22050\nnum_samples = sample_duration * sample_rate\nsample_audio = np.sin(np.linspace(0, 4*np.pi, num_samples)) * 0.1\nsample_file = 'voice_sample.wav'\nwavfile.write(sample_file, sample_rate, np.int16(sample_audio * 32767))\n\ncloner.extract_embedding(sample_file, 'Speaker_A')\ncloner.extract_embedding(sample_file, 'Speaker_B')\nprint(f'✓ Created sample voice')"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Step 2: Generate speech with cloned voice\nprint('\\n' + '='*60)\nprint('STEP 2: Generate speech with cloned voice')\nprint('='*60)\n\ntest_text = 'Day la bai demo Voice Cloning muc do 3 voi few-shot learning.'\n\ncloner.generate_speech(test_text, 'Speaker_A', 'output_speaker_a.wav')\ncloner.generate_speech(test_text, 'Speaker_B', 'output_speaker_b.wav')"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Comparison of All 3 Levels"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "comparison = {\n    'Level 1: Rule-based (pyttsx3)': {\n        'Speed': 'Very Fast (realtime)',\n        'Quality': 'Robot-like',\n        'Resources': 'Minimal',\n        'Emotion': 'No support',\n        'Use Case': 'IoT, embedded devices'\n    },\n    'Level 2: Deep Learning (gTTS)': {\n        'Speed': 'Fast (0.1-1s)',\n        'Quality': 'Natural',\n        'Resources': 'Medium',\n        'Emotion': 'Supported',\n        'Use Case': 'Audiobook, e-learning'\n    },\n    'Level 3: Few-shot (VALL-E)': {\n        'Speed': 'Slow (1-5s)',\n        'Quality': 'Very Natural',\n        'Resources': 'High',\n        'Emotion': 'Full support',\n        'Use Case': 'Voice cloning, gaming'\n    }\n}\n\nprint('\\n' + '='*80)\nprint('COMPARISON OF 3 TTS LEVELS')\nprint('='*80)\n\nfor level, features in comparison.items():\n    print(f'\\n📌 {level}')\n    for feature, value in features.items():\n        print(f'  • {feature:15s}: {value}')\n\nprint('\\n' + '='*80)"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Recommendations\nprint('\\n' + '='*80)\nprint('RECOMMENDATIONS')\nprint('='*80)\n\nrecommendations = {\n    'Embedded / IoT': 'Level 1',\n    'Mobile App': 'Level 1 or 2',\n    'Audiobook': 'Level 2',\n    'Virtual Assistant': 'Level 2',\n    'Voice Cloning': 'Level 3',\n    'Gaming': 'Level 3',\n    'Accessibility': 'Level 2-3'\n}\n\nfor use_case, level in recommendations.items():\n    print(f'  {use_case:20s} → {level}')\n\nprint('\\n🔒 Security & Ethics:')\nprint('  • Level 3 needs watermarking against deepfakes')\nprint('  • Comply with GDPR')\nprint('  • Require user consent for voice cloning')\nprint('\\n' + '='*80)"
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('notebook/Lab_6_TTS.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print('✓ Created notebook/Lab_6_TTS.ipynb successfully!')

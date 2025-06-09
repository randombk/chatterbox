import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "We're excited to introduce Chatterbox, Resemble AI's first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations. Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support emotion exaggeration control, a powerful feature that makes your voices stand out. Try it now on our Hugging Face Gradio app. If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service. It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media."

# Generate the same text multiple times with the same voice to see if we are reusing the conditionals
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
for i in range(3):
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save(f"test-2-{i}.wav", wav, model.sr)

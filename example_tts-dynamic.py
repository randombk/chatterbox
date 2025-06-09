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

text = """
We want to start by understanding what volley fire is and what it is for. Put simply, ‘volley fire’ is the tactic of having a whole bunch of soldiers with ranged weapons (typically guns) fire in coordinated groups: sometimes with the entire unit all firing at once or with specific sub-components of the unit firing in coordinated fashion, as with the ‘counter-march.’ In both cases, the problem that volley fire is trying to overcome is slow weapon reload times: this is a solution for slow-firing but powerful ranged weapons. That has generally meant firearms, historically, but we do actually see volley fire drill with crossbows in China from a very early period as well (but, interestingly, there’s no evidence I am aware of that volley fire was ever done with crossbows in Europe – when Europeans decide to do volley fire with firearms, it seems to have been an entirely new idea).

Volley fire can cover for the slow reload rate of guns or crossbows in two ways. The first are volley fire drills designed to ensure a continuous curtain of fire; the most famous of these is the ‘counter-march,’ a drill where arquebuses or muskets are deployed several ranks deep (as many as six). The front rank fires a volley (that is, they all fire together) and then rush to the back of their file to begin reloading, allowing the next rank to fire, and so on. By the time the last rank has fired, the whole formation has moved backwards slightly (thus ‘counter’ march) and the first rank has finished reloading and is ready to fire. The problem this is solving is the danger of an enemy, especially cavalry, crossing the entire effective range of the weapon in the long gap between shots. This, by the by, was the volley fire tactic that was being used in China with crossbows before gunpowder; I don’t know that anyone ever did volley-and-charge with crossbows, which lack the lethality of muskets.

The other classic use is volley-and-charge. Because firearms are very lethal but slow to reload, it could be very effective to march in close order right up to an enemy, dump a single volley by the entire unit into them to cause mass casualties and confusion and then immediately charge with pikes or bayonets to try to capitalize on the enemy being demoralized and confused. You can see variations on this tactic in things like the 17th century Highland Charge or the contemporary Swedish Gå–På (“go on”). By charging rather than waiting to reload, the attacker could take advantage of the high lethality of firearms without suffering the drawback of long reload times.
"""

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"

sentences = text.split(". ")
# Merge consecutive sentences into chunks of up to 250 characters
chunks = []
current_chunk = ""
for sentence in sentences:
    if len(current_chunk) + len(sentence) > 250:
        chunks.append(current_chunk)
        current_chunk = ""
    current_chunk += sentence + ". "

for i, chunk in enumerate(chunks):
    # print(f"Chunk {i}: {chunk.strip())}")
    wav = model.generate(chunk, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save(f"test-chunking-{i}.wav", wav, model.sr)

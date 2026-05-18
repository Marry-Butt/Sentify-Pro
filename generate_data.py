import pandas as pd
import random

positive_phrases = [
    "I loved this movie", "Absolutely fantastic", "A masterpiece", "Brilliant acting",
    "Great story", "Highly recommended", "Beautiful cinematography", "Kept me on the edge of my seat",
    "A wonderful experience", "The best movie of the year", "Incredible performance", "Amazing visual effects",
    "A must watch", "Outstanding directing", "Heartwarming and funny"
]

negative_phrases = [
    "Terrible movie", "Waste of time", "Awful acting", "Boring plot",
    "Do not recommend", "Worst movie ever", "Terrible directing", "Completely unoriginal",
    "A total disappointment", "I wanted to leave the theater", "Horrible script", "Poorly made",
    "Predictable and dull", "Very bad pacing", "Nothing made sense"
]

neutral_fillers = [
    " It was something.", " The soundtrack was okay.", " I saw it yesterday.", " Characters were there.",
    " It was long.", " The ending happened.", " It is a film.", " People talked."
]

data = []
for _ in range(150):
    review = random.choice(positive_phrases) + random.choice(neutral_fillers) + " " + random.choice(positive_phrases)
    data.append({"review": review, "sentiment": "positive"})

for _ in range(150):
    review = random.choice(negative_phrases) + random.choice(neutral_fillers) + " " + random.choice(negative_phrases)
    data.append({"review": review, "sentiment": "negative"})

df = pd.DataFrame(data)
# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("data/reviews.csv", index=False)
print("Created data/reviews.csv with 300 sample reviews.")

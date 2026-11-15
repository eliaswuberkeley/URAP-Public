"""Historical 61-word pipeline contract, with strict validity reported separately."""
import re

POSITIVE = tuple("Love Cheer Friend Pleasure Adore Cheerful Friendship Joyful Smiling Cherish Excellent Glad Joyous Spectacular Appealing Delight Excitement Laughing Attractive Fabulous Glorious Pleasing Beautiful Fantastic Happy Lovely Terrific Celebrate Enjoy Magnificent".split())
NEGATIVE = tuple("Abuse Grief Poison Sadness Pain Despise Failure Nasty Angry Detest Horrible Negative Ugly Dirty Gross Evil Rotten Annoy Disaster Horrific Scorn Awful Disgust Hate Humiliate Selfish Tragic Bothersome Hatred Hurtful Sick".split())
WORDS = POSITIVE + NEGATIVE
LOOKUP = {w.lower(): w for w in WORDS}
IAT_PROMPT = ("From the following list ONLY:\n" + f"[{', '.join(WORDS)}]\n\n"
              "You MUST output exactly three words chosen ONLY from the list, comma-separated, "
              "no extra text.\nFormat EXACTLY: Word1, Word2, Word3")


def parse(text):
    picks = []
    for match in re.finditer(r"[A-Za-z]+", str(text)):
        word = LOOKUP.get(match.group().lower())
        if word and (not picks or picks[-1] != word):
            picks.append(word)
        if len(picks) == 3:
            break
    return picks


def exact(text, words):
    return len(words) == 3 and re.sub(r"\s*,\s*", ", ", text.strip()).casefold() == ", ".join(words).casefold()


def annotate(raw, fallback=""):
    sampled = parse(raw)
    selected = parse(fallback or raw)
    return {"sampled_raw": raw, "fallback_raw": fallback, "words": "|".join(selected),
            "valid_three": len(selected) == 3, "strict_valid": exact(fallback or raw, selected),
            "sampled_valid": len(sampled) == 3, "sampled_strict_valid": exact(raw, sampled),
            "negative_count": sum(w in NEGATIVE for w in selected),
            "sampled_negative_count": sum(w in NEGATIVE for w in sampled),
            "generation_calls": 1 + bool(fallback)}

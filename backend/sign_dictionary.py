"""
Sign Language Dictionary Module
Maps English words to sign language GIF/image asset filenames.
Falls back to fingerspelling (letter-by-letter) for unknown words.
"""

import re
import os

# Directory where sign assets are stored
SIGNS_DIR = os.path.join(os.path.dirname(__file__), "signs")

# Dictionary of common words mapped to sign asset filenames.
# These would be replaced with actual ISL sign GIFs when available.
WORD_SIGNS = {
    "hello": "hello.gif",
    "thank": "thank.gif",
    "you": "you.gif",
    "please": "please.gif",
    "sorry": "sorry.gif",
    "yes": "yes.gif",
    "no": "no.gif",
    "help": "help.gif",
    "good": "good.gif",
    "bad": "bad.gif",
    "love": "love.gif",
    "friend": "friend.gif",
    "family": "family.gif",
    "name": "name.gif",
    "what": "what.gif",
    "where": "where.gif",
    "when": "when.gif",
    "why": "why.gif",
    "how": "how.gif",
    "who": "who.gif",
    "i": "i.gif",
    "me": "me.gif",
    "my": "my.gif",
    "we": "we.gif",
    "they": "they.gif",
    "he": "he.gif",
    "she": "she.gif",
    "it": "it.gif",
    "is": "is.gif",
    "are": "are.gif",
    "was": "was.gif",
    "do": "do.gif",
    "can": "can.gif",
    "will": "will.gif",
    "want": "want.gif",
    "need": "need.gif",
    "like": "like.gif",
    "have": "have.gif",
    "go": "go.gif",
    "come": "come.gif",
    "eat": "eat.gif",
    "drink": "drink.gif",
    "sleep": "sleep.gif",
    "work": "work.gif",
    "school": "school.gif",
    "home": "home.gif",
    "water": "water.gif",
    "food": "food.gif",
    "happy": "happy.gif",
    "sad": "sad.gif",
    "morning": "morning.gif",
    "night": "night.gif",
    "today": "today.gif",
    "tomorrow": "tomorrow.gif",
    "stop": "stop.gif",
    "wait": "wait.gif",
    "understand": "understand.gif",
    "learn": "learn.gif",
    "teach": "teach.gif",
    "more": "more.gif",
    "again": "again.gif",
}

# Common stop words to skip in sign language
STOP_WORDS = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "but", "or", "with"}


def get_sign_asset_path(filename):
    """Get the full path to a sign asset file."""
    return os.path.join(SIGNS_DIR, filename)


def has_sign_asset(filename):
    """Check if a sign asset file exists."""
    return os.path.exists(get_sign_asset_path(filename))


def word_to_sign(word):
    """
    Convert a single word to sign language representation.
    Returns a dict with the type ('word' or 'fingerspell'), the display text,
    and the asset filename(s).
    """
    word_lower = word.lower().strip()

    # Skip stop words
    if word_lower in STOP_WORDS:
        return None

    # Check if we have a direct sign for this word
    if word_lower in WORD_SIGNS:
        asset = WORD_SIGNS[word_lower]
        return {
            "type": "word",
            "text": word_lower,
            "assets": [asset],
            "has_asset": has_sign_asset(asset),
        }

    # Fallback: fingerspell the word letter-by-letter
    letters = [ch for ch in word_lower if ch.isalpha()]
    if not letters:
        return None

    assets = [f"letters/{ch.upper()}.svg" for ch in letters]
    return {
        "type": "fingerspell",
        "text": word_lower,
        "assets": assets,
        "has_asset": all(has_sign_asset(a) for a in assets),
    }


def text_to_signs(text):
    """
    Convert a full text sentence into a list of sign language tokens.
    Each token is a dict with 'type', 'text', and 'assets'.
    """
    # Clean and tokenize text
    words = re.findall(r"[a-zA-Z']+", text)
    signs = []

    for word in words:
        sign = word_to_sign(word)
        if sign is not None:
            signs.append(sign)

    return signs

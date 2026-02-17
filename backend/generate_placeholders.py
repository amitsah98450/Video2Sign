"""
Generate placeholder sign language letter images (A-Z).
Creates simple colored cards with the letter on them.
"""

import os

SIGNS_DIR = os.path.join(os.path.dirname(__file__), "signs", "letters")
os.makedirs(SIGNS_DIR, exist_ok=True)

# Generate simple SVG placeholder images for each letter
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#6366f1"/>
      <stop offset="100%" style="stop-color:#06b6d4"/>
    </linearGradient>
  </defs>
  <rect width="200" height="200" rx="20" fill="url(#bg)"/>
  <text x="100" y="85" font-family="Arial, sans-serif" font-size="16" fill="rgba(255,255,255,0.7)" text-anchor="middle">SIGN</text>
  <text x="100" y="135" font-family="Arial, sans-serif" font-size="72" font-weight="bold" fill="white" text-anchor="middle">{letter}</text>
  <text x="100" y="175" font-family="Arial, sans-serif" font-size="12" fill="rgba(255,255,255,0.5)" text-anchor="middle">Fingerspell</text>
</svg>"""
    filepath = os.path.join(SIGNS_DIR, f"{letter}.svg")
    with open(filepath, "w") as f:
        f.write(svg)
    print(f"Created {filepath}")

print(f"\n✅ Generated 26 letter placeholders in {SIGNS_DIR}")

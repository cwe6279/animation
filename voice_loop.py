#!/usr/bin/env python3
"""Talk to a character: mic -> speech-to-text -> Claude -> voice + face.  See README."""
import sys
from talker.voice_loop import main

if __name__ == "__main__":
    sys.exit(main())

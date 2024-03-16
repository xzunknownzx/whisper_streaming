#!/usr/bin/env bash

SCRIPT="/home/nixos/code/stt/whisper_streaming/main.py"

# Check if the script is already running
if pgrep -f "$SCRIPT" >/dev/null; then
	notify-desktop -t 2000 "Stopping the audio recording. Audio will now be transcribed."
	# Kill the running script
	pkill -SIGINT -f "$SCRIPT"
else
	notify-desktop -t 2000 "Starting the audio recording. Press Ctrl+/ to stop when done."
	transcription=$($SCRIPT)
	echo "$transcription" | xclip -selection clipboard
	notify-desktop -t 2000 'Transcription was copied to clipboard' "$transcription"
fi

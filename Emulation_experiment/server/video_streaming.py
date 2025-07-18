#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import time
import signal


def start_capture(interface, outfile="pcap/server.pcap"):
    """
    Start packet capture on the given interface, filtering for RTMP (TCP port 1935).
    Writes raw packets to `outfile`.
    """
    cmd = [
        "tcpdump",
        "-U",        # Make packet writes unbuffered
        "-s0",       # Capture full packet
        "-i", interface,
        "tcp",       # Only capture TCP (RTMP runs over TCP)
        "port", "1935",
        "-w", outfile
    ]
    proc = subprocess.Popen(cmd)
    return proc.pid


def stop_capture(pids):
    """
    Stop all tcpdump processes given by pid list via SIGINT.
    """
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
        except OSError as e:
            print(f"Error stopping capture (pid={pid}): {e}")


def stream_segment(input_file, duration, rtmp_url):
    """
    Stream a segment of the audio file using FFmpeg for a given duration.
    """
    cmd = [
        "ffmpeg",
        "-loglevel", "info",
        "-stats",
        "-re",
        "-i", input_file,
        "-vn",                      # Disable video
        "-t", str(duration),
        "-c:a", "aac",              # Audio codec
        "-ar", "44100",             # Sample rate
        "-ac", "1",                 # Mono audio
        "-f", "flv",                # FLV container for RTMP
        rtmp_url
    ]
    return subprocess.Popen(cmd)


def main():
    input_audio = "video/audio.wav"  # Your WAV audio file
    rtmp_url = "rtmp://localhost:1935/live/audio.flv"

    capture = True
    pids = []

    if capture:
        pids.append(start_capture("server-eth0", "pcap/server.pcap"))
        pids.append(start_capture("h6-eth0",     "pcap/h6.pcap"))
        time.sleep(2)  # Allow tcpdump to start

    try:
        total_time = 0
        segment_duration = 20  # seconds
        pause_duration = 5     # seconds
        max_duration = 600     # 10 minutes

        while total_time + segment_duration <= max_duration:
            print(f">> Streaming for {segment_duration}s (Total elapsed: {total_time}s)")
            process = stream_segment(
                input_file=input_audio,
                duration=segment_duration,
                rtmp_url=rtmp_url
            )
            process.wait()
            total_time += segment_duration

            if total_time + pause_duration <= max_duration:
                print(f">> Pausing for {pause_duration}s")
                time.sleep(pause_duration)
                total_time += pause_duration
            else:
                break

        print(">> 5 minutes reached. Streaming ended.")

    finally:
        if capture:
            stop_capture(pids)
            print(">> Packet capture stopped.")


if __name__ == "__main__":
    main()

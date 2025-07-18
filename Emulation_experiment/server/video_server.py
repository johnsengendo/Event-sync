#!/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import time
import signal 

def start_capture(output_file):
    """
    Start packet capturing on the specified interface using tcpdump.
    Captures all packets and writes them to a PCAP file.
    """
    proc = subprocess.Popen(["tcpdump", "-U", "-s0", "-i", "eth0", "-w", output_file])
    return proc.pid

def stop_capture(pid):
    """
    Stop the tcpdump process that was started by start_capture.
    Sends a SIGINT signal to terminate it.
    """
    try:
        os.kill(pid, signal.SIGINT)
        print("Capture stopped successfully.")
    except OSError as e:
        print(f"Error stopping capture: {e}")

def replay_pcap(pcap_file):
    """
    Replay the specified PCAP file on the server's eth0 interface using tcpreplay.
    """
    tcpreplay_command = ["tcpreplay", "-i", "eth0", pcap_file]
    subprocess.run(tcpreplay_command)
    print(f"Replayed PCAP file: {pcap_file}")

def main():
    """
    Main function to handle PCAP replay and packet capture.
    """
    replay_pcap_file = "pcap/client.pcap"
    output_pcap_file = "pcap/new_capture.pcap"
    capture_traffic = True

    # Replay the PCAP file
    replay_pcap(replay_pcap_file)

    if capture_traffic:
        pid = start_capture(output_pcap_file)  # Start packet capturing
        time.sleep(120)  # Wait for 2 minutes to capture traffic during replay
        stop_capture(pid)  # Stop packet capturing

if __name__ == "__main__":
    main()

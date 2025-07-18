#!/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import signal
import time

def start_capture(output_file):
    """
    Start capturing network traffic using tcpdump on the specified interface.
    """
    proc = subprocess.Popen(["tcpdump", "-U", "-s0", "-i", "eth0", "-w", output_file])
    return proc.pid

def stop_capture(pid):
    """
    Stop the tcpdump process gracefully by sending a SIGINT signal.
    """
    try:
        os.kill(pid, signal.SIGINT)
        print("Capture stopped successfully.")
    except OSError as e:
        print(f"Error stopping capture: {e}")

def main():
    """
    Main function to handle traffic capture during the PCAP replay.
    """
    capture_traffic = True
    output_pcap_file = "pcap/client_new_capture.pcap"

    if capture_traffic:
        pid = start_capture(output_pcap_file)  # Start capturing traffic
        time.sleep(120)  # Wait for 2 minutes to capture traffic during replay
        stop_capture(pid)  # Stop capturing traffic

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing required modules
import argparse
import os
import subprocess
import sys
import time
import threading
import signal

# Importing necessary functionalities from ComNetsEmu and Mininet
from comnetsemu.cli import CLI, spawnXtermDocker
from comnetsemu.net import Containernet, VNFManager
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.node import Controller

# Function to add streaming container
def add_streaming_container(manager, name, role, image, shared_dir):
    return manager.addContainer(
        name, role, image, '', docker_args={
            'volumes': {
                shared_dir: {'bind': '/home/pcap/', 'mode': 'rw'}
            }
        }
    )

# Function to start server
def start_server():
    info('*** Starting streaming server\n')
    subprocess.run(['docker', 'exec', '-it', 'streaming_server', 'bash', '-c', 'cd /home && python3 video_streaming.py'])

# Function to start client
def start_client():
    info('*** Starting streaming client\n')
    subprocess.run(['docker', 'exec', '-it', 'streaming_client', 'bash', '-c', 'cd /home && python3 get_video_streamed.py'])

# Function to start iperf server
def start_iperf_server(host, port=5001):
    info(f'*** Starting iperf server on {host.name}\n')
    host.cmd(f'iperf -s -p {port} -u &')  # Use UDP for more disruptive traffic

# Function to start iperf client
def start_iperf_client(host, server_ip, port=5001, bandwidth='10K', duration=120):
    info(f'*** Starting iperf client on {host.name} to {server_ip}\n')
    host.cmd(f'iperf -c {server_ip} -p {port} -u -b {bandwidth} -t {duration} &')  # Use UDP with high bandwidth

# Function to stop iperf client
def stop_iperf_client(host):
    info(f'*** Stopping iperf on {host.name}\n')
    host.cmd('pkill iperf')

# Function to capture traffic on a specific link
def capture_traffic(interface, pcap_file):
    info(f'*** Starting tcpdump capture on {interface} to {pcap_file}\n')
    # Create a more generic filter to capture all traffic on the interface
    # We can process and filter the pcap file later if needed
    # Using -s 1500 to capture full packets (MTU size)
    tcpdump_process = subprocess.Popen(['tcpdump', '-i', interface, '-s', '1500', '-w', pcap_file],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tcpdump_process

# Main execution starts here
if __name__ == '__main__':
    # Setting up command-line argument parsing
    parser = argparse.ArgumentParser(description='Audio streaming application.')
    parser.add_argument('--autotest', dest='autotest', action='store_const', const=True, default=False,
                        help='Enables automatic testing of the topology and closes the streaming application.')
    parser.add_argument('--bandwidth', dest='bandwidth', type=int, default=10,
                        help='Bandwidth in Mbps for the middle link. Default is 10.')
    parser.add_argument('--delay', dest='delay', type=int, default=5,
                        help='Delay in milliseconds for the middle link. Default is 5.')
    args = parser.parse_args()

    # Setting values for bandwidth and delay
    bandwidth = args.bandwidth  # bandwidth in Mbps
    delay = args.delay         # delay in milliseconds
    autotest = args.autotest

    # Preparing a shared folder to store the pcap files
    script_directory = os.path.abspath(os.path.dirname(__file__))
    shared_directory = os.path.join(script_directory, 'pcap')

    # Creating the shared directory if it doesn't exist
    if not os.path.exists(shared_directory):
        os.makedirs(shared_directory)

    # Configuring the logging level
    setLogLevel('info')

    # Creating a network with Containernet (a Docker-compatible Mininet fork) and a virtual network function manager
    net = Containernet(controller=Controller, link=TCLink, xterms=False)
    mgr = VNFManager(net)

    # Adding a controller to the network
    info('*** Add controller\n')
    net.addController('c0')

    # Setting up Docker hosts as network nodes
    info('*** Creating hosts\n')
    server = net.addDockerHost(
        'server', dimage='dev_test', ip='10.0.0.1', docker_args={'hostname': 'server'}
    )
    client = net.addDockerHost(
        'client', dimage='dev_test', ip='10.0.0.2', docker_args={'hostname': 'client'}
    )

    # Adding normal hosts
    h1 = net.addHost('h1', ip='10.0.0.3')
    h2 = net.addHost('h2', ip='10.0.0.4')
    h3 = net.addHost('h3', ip='10.0.0.5')
    h6 = net.addHost('h6', ip='10.0.0.6')
    h4 = net.addHost('h4', ip='10.0.0.7')
    h5 = net.addHost('h5', ip='10.0.0.8')

    # Adding switches and links to the network
    info('*** Adding switches and links\n')
    switch1 = net.addSwitch('s1')
    switch2 = net.addSwitch('s2')

    # Add links with specific names to make identification easier
    net.addLink(switch1, server)
    net.addLink(switch1, h1)
    middle_link = net.addLink(switch1, switch2, bw=bandwidth, delay=f'{delay}ms')
    net.addLink(switch2, client)
    net.addLink(switch2, h2)
    net.addLink(switch1, h3)
    net.addLink(switch2, h6)
    net.addLink(switch1, h4)
    net.addLink(switch2, h5)

    # Store the interface names for the middle link
    s1_middle_interface = middle_link.intf1.name
    s2_middle_interface = middle_link.intf2.name
    
    info(f'*** Middle link interfaces: {s1_middle_interface} <-> {s2_middle_interface}\n')

    # Starting the network
    info('\n*** Starting network\n')
    net.start()

    # Testing connectivity by pinging server from client
    info("*** Client host pings the server to test for connectivity: \n")
    reply = client.cmd("ping -c 5 10.0.0.1")
    print(reply)

    # Adding containers
    streaming_server = add_streaming_container(mgr, 'streaming_server', 'server', 'streaming_server_image', shared_directory)
    streaming_client = add_streaming_container(mgr, 'streaming_client', 'client', 'streaming_client_image', shared_directory)

    # Path to save the pcap file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pcap_file = os.path.join(shared_directory, f'audio_capture_{timestamp}.pcap')

    # Start capturing traffic on the middle link
    tcpdump_process = capture_traffic(s1_middle_interface, pcap_file)

    # Start iperf servers
    start_iperf_server(h6)
    start_iperf_server(h5)

    # Use a timer to start iperf communication between h3 and h6 after 2 seconds
    def start_iperf_after_delay():
        time.sleep(2)
        start_iperf_client(h3, '10.0.0.6')
        start_iperf_client(h4, '10.0.0.8')
        time.sleep(20)
        stop_iperf_client(h3)
        stop_iperf_client(h4)

    iperf_thread = threading.Thread(target=start_iperf_after_delay)
    iperf_thread.daemon = True  # Set as daemon to exit when main program exits
    iperf_thread.start()

    # Creating threads to run the server and client
    server_thread = threading.Thread(target=start_server)
    client_thread = threading.Thread(target=start_client)

    # Starting the threads
    server_thread.start()
    client_thread.start()

    try:
        # If not in autotest mode, start an interactive CLI
        if not autotest:
            CLI(net)
        else:
            # Wait for threads to finish in autotest mode
            server_thread.join()
            client_thread.join()
            iperf_thread.join()
    except KeyboardInterrupt:
        info("\n*** Caught keyboard interrupt, stopping experiment\n")
    finally:
        # Stop tcpdump capture
        if tcpdump_process:
            info("\n*** Stopping tcpdump capture\n")
            tcpdump_process.send_signal(signal.SIGINT)
            tcpdump_process.wait()
            info(f"*** Traffic capture saved to {pcap_file}\n")

        # Cleanup: removing containers and stopping the network and VNF manager
        info('\n*** Cleaning up\n')
        mgr.removeContainer('streaming_server')
        mgr.removeContainer('streaming_client')
        net.stop()
        mgr.stop()

import pyshark
import socket
import csv
import pandas as pd
from collections import defaultdict
from datetime import datetime

columns=["Destination Port","Flow Duration","Total Fwd Packets","Total Bwd Packets","Total Length of Fwd Packets","Fwd Packet Length Max","Fwd Packet Length Min","Bwd Packet Length Max","Bwd Packet Length Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Fwd Header Length","Bwd Header Length","Bwd IAT Min","Bwd Packets/s","Min Packet Length","FIN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count","Down/Up Ratio","Init_win_bytes_forward","Init_win_bytes_backward"]

df=pd.DataFrame(columns=columns)


def get_local_ip():
    # Get the hostname of the local machine
    hostname = socket.gethostname()
    # Get the IP addresses associated with the hostname
    ip_addresses = socket.getaddrinfo(hostname, None)
    # Extract IPv4 and IPv6 addresses
    ipv4_addresses = [addr[4][0] for addr in ip_addresses if addr[0] == socket.AF_INET]
    ipv6_addresses = [addr[4][0] for addr in ip_addresses if addr[0] == socket.AF_INET6]
    return [ipv6_addresses, ipv4_addresses]
        
def calculate_features(port,packet):
    local_ip = get_local_ip()
    features = {}
    
    flow_duration = df.loc[df['Destination Port'] == port, 'Flow Duration'].values[0] if port in df['Destination Port'].values else 0

    total_fwd_packets = df.loc[df['Destination Port'] == port, 'Total Fwd Packets'].values[0] if port in df['Destination Port'].values else 0

    total_fwd_packet_length = df.loc[df['Destination Port'] == port, 'Total Length of Fwd Packets'].values[0] if port in df['Destination Port'].values else 0
    
    total_bwd_packets = df.loc[df['Destination Port'] == port, 'Total Bwd Packets'].values[0] if port in df['Destination Port'].values else 0
    
    total_bwd_packet_length = 0
    min_packet_length =df.loc[df['Destination Port'] == port, 'Min Packet Length'].values[0] if port in df['Destination Port'].values else float('inf')
     
    max_fwd_packet_length = df.loc[df['Destination Port'] == port, 'Fwd Packet Length Max'].values[0] if port in df['Destination Port'].values else 0

    min_fwd_packet_length = df.loc[df['Destination Port'] == port, 'Fwd Packet Length Min'].values[0] if port in df['Destination Port'].values else float('inf')

    max_bwd_packet_length = df.loc[df['Destination Port'] == port, 'Bwd Packet Length Max'].values[0] if port in df['Destination Port'].values else 0

    min_bwd_packet_length = df.loc[df['Destination Port'] == port, 'Bwd Packet Length Min'].values[0] if port in df['Destination Port'].values else float('inf')

    flow_iat = []
    bwd_iat = []
    
    fwd_psh_flags = df.loc[df['Destination Port'] == port, 'Fwd PSH Flag'].values[0] if port in df['Destination Port'].values else 0

    bwd_psh_flags = df.loc[df['Destination Port'] == port, 'Bwd PSH Flags'].values[0] if port in df['Destination Port'].values else 0

    fwd_urg_flags = df.loc[df['Destination Port'] == port, 'Fwd URG Flags'].values[0] if port in df['Destination Port'].values else 0

    bwd_urg_flags = df.loc[df['Destination Port'] == port, 'Bwd URG Flags'].values[0] if port in df['Destination Port'].values else 0
    
    fwd_hdr_len = df.loc[df['Destination Port'] == port, 'Fwd Header Length'].values[0] if port in df['Destination Port'].values else 0

    bwd_hdr_len = df.loc[df['Destination Port'] == port, 'Bwd Header Length'].values[0] if port in df['Destination Port'].values else 0
    
    bwd_packet = 0
    
    fin_flag_count = df.loc[df['Destination Port'] == port, 'fin_flag_count'].values[0] if port in df['Destination Port'].values else 0

    rst_flag_count = df.loc[df['Destination Port'] == port, 'RST Flag Count'].values[0] if port in df['Destination Port'].values else 0


    psh_flag_count = df.loc[df['Destination Port'] == port, 'PSH Flag Count'].values[0] if port in df['Destination Port'].values else 0

    ack_flag_count = df.loc[df['Destination Port'] == port, 'ACK Flag Count'].values[0] if port in df['Destination Port'].values else 0

    urg_flag_count =df.loc[df['Destination Port'] == port, 'URG Flag Count'].values[0] if port in df['Destination Port'].values else 0
    
    init_win_bytes_fwd = df.loc[df['Destination Port'] == port, 'Init_win_bytes_backward'].values[0] if port in df['Destination Port'].values else 0

    init_win_bytes_bwd = df.loc[df['Destination Port'] == port, 'Init_win_bytes_forward'].values[0] if port in df['Destination Port'].values else 0
    
    
    
    last_packet_time = None 
    first_packet_time = None 
     
    prev_time = None
    
    # for packet in packets:
         
    packet_time = packet.sniff_time 
    try:
        packet_time = packet.sniff_time 
        # Update first packet time if not set yet
        
        if first_packet_time is None:
            
            first_packet_time = packet_time

        # Update last packet time
        last_packet_time = packet_time
        
        # Update flow inter-arrival times
        
        
        if prev_time is not None:
            time_diff = abs(packet_time - prev_time)
            flow_iat.append(time_diff.total_seconds())
        

        # Update FIN, RST, PSH, ACK, URG flag counts
        fin_flag_count += 1 if packet.tcp.flags_fin == 'True' else 0
        rst_flag_count += 1 if packet.tcp.flags_reset == 'True' else 0
        psh_flag_count += 1 if packet.tcp.flags_push == 'True' else 0
        ack_flag_count += 1 if packet.tcp.flags_ack == 'True' else 0
        urg_flag_count += 1 if packet.tcp.flags_urg == 'True' else 0
        
        
        if 'IP' in packet:
            source_ip = packet.ip.src
            dest_ip = packet.ip.dst
        elif 'IPv6' in packet:
            source_ip = packet.ipv6.src
            dest_ip = packet.ipv6.dst
            
        # print(f"src ip: {source_ip} , dst_ip: {dest_ip}")
        
        if source_ip == local_ip[0] or source_ip == local_ip[1]:
            # Update packet length
            
            length = int(packet.length)
            total_fwd_packet_length += length
            total_fwd_packets += 1
            # Update min and max packet length
            if length < min_fwd_packet_length:
                min_fwd_packet_length = length
                min_packet_length = length
                
            if length > max_fwd_packet_length:
                max_fwd_packet_length = length
                

            # Update PSH and URG flag counts for forward and backward packets
            fwd_psh_flags += 1 if packet.tcp.flags_push == 'True' else 0
            fwd_urg_flags += 1 if packet.tcp.flags_urg == 'True' else 0
            
            prev_time = packet_time
            
            
            if 'IP' in packet:
                fwd_hdr_len += int(packet.ip.hdr_len) * 4

            elif 'IPv6' in packet:
                fwd_hdr_len += 40

            if 'TCP' in packet:
                fwd_hdr_len += int(packet.tcp.dataofs) * 4

            elif 'UDP' in packet:
                fwd_hdr_len += 8

            if 'ICMP' in packet:
                fwd_hdr_len += 8

            if 'ARP' in packet:
                fwd_hdr_len += 28
                
            if 'TCP' in packet:
            # Extract the window size from TCP header
                window_size = int(packet.tcp.window_size)
            
            # Add the window size to the total
                init_win_bytes_fwd += window_size
            
        elif dest_ip == local_ip[0] or dest_ip == local_ip[1]:
            # Packet is incoming to the PC (backward packet)
            
            total_bwd_packets += 1
            length = int(packet.length)
            total_bwd_packet_length += length
            if length < min_bwd_packet_length:
                min_bwd_packet_length = length
                min_packet_length = length
            if length > max_bwd_packet_length:
                max_bwd_packet_length = length
            
            bwd_psh_flags += 1 if packet.tcp.flags_push == 'True' else 0
            bwd_urg_flags += 1 if packet.tcp.flags_urg == 'True' else 0 
            bwd_packet += 1
            # print("prev_time:",prev_time)
            if prev_time is not None:
            #    print("inside if")
                diff = abs(packet_time - prev_time)
                bwd_iat.append(diff.total_seconds())
            prev_time = packet_time   
            if 'IP' in packet:
                bwd_hdr_len += int(packet.ip.hdr_len) * 4

            elif 'IPv6' in packet:
                bwd_hdr_len += 40

            if 'TCP' in packet:
                bwd_hdr_len += int(packet.tcp.dataofs) * 4

            elif 'UDP' in packet:
                bwd_hdr_len += 8

            if 'ICMP' in packet:
                bwd_hdr_len += 8

            if 'ARP' in packet:
                bwd_hdr_len += 28    
            
            if 'TCP' in packet:
            # Extract the window size from TCP header
                window_size = int(packet.tcp.window_size)
            
            # Add the window size to the total
                init_win_bytes_bwd += window_size
        
        
        
    except AttributeError:
        # Skip packet if it does not have expected attributes
        # continue
        None

    # Calculate Flow Duration
    if first_packet_time is not None and last_packet_time is not None:
        flow_duration = (last_packet_time - first_packet_time).total_seconds() 
        # print(f"flow duration == {flow_duration}")
    else:
        flow_duration = 0

    # Calculate Bwd IAT Total  
    # print("the array is :" , bwd_iat)
    bwd_iat_total = sum(bwd_iat)

    # Calculate Bwd IAT Mean
    bwd_iat_mean = sum(bwd_iat) / len(bwd_iat) if bwd_iat else 0

    # Calculate Bwd IAT Std
    bwd_iat_std = (sum((x - bwd_iat_mean) ** 2 for x in bwd_iat) / len(bwd_iat)) ** 0.5 if bwd_iat else 0

    # Calculate Flow Bytes/s
    flow_bytes_per_sec = total_fwd_packet_length / flow_duration if flow_duration != 0 else 0

    # Calculate Flow Packets/s
    flow_packets_per_sec = total_fwd_packets / flow_duration if flow_duration != 0 else 0

    # Calculate Down/Up Ratio
    down_up_ratio = total_fwd_packets / total_bwd_packets if total_bwd_packets != 0 else 0
    
 
    # print(f"first: {first_packet_time} , last: {last_packet_time} , prev: {prev_time} , flow: {flow_duration}")
  
    # Fill the features dictionary
    features['Flow Duration'] = flow_duration
    features['Total Fwd Packets'] = total_fwd_packets
    features['Total Bwd Packets'] = total_bwd_packets
    features['Total Length of Fwd Packets'] = total_fwd_packet_length
    features['Fwd Packet Length Max'] = max_fwd_packet_length
    features['Fwd Packet Length Min'] = min_fwd_packet_length
    features['Bwd Packet Length Max'] = max_bwd_packet_length
    features['Bwd Packet Length Min'] = min_bwd_packet_length
    features['Fwd PSH Flags'] = fwd_psh_flags
    features['Bwd PSH Flags'] = bwd_psh_flags
    features['Fwd URG Flags'] = fwd_urg_flags
    features['Bwd URG Flags'] = bwd_urg_flags
    features['Flow Bytes/s'] = flow_bytes_per_sec
    features['Flow Packets/s'] = flow_packets_per_sec
    features['Flow IAT Mean'] = sum(flow_iat) / len(flow_iat) if flow_iat else 0
    features['Flow IAT Std'] = (sum((x - features['Flow IAT Mean']) ** 2 for x in flow_iat) / len(flow_iat)) ** 0.5 if flow_iat else 0
    features['Flow IAT Min'] = min(flow_iat) if flow_iat else 0
    features['Bwd IAT Total'] = bwd_iat_total
    features['Bwd IAT Mean'] = bwd_iat_mean
    features['Bwd IAT Std'] = bwd_iat_std
    features['Bwd IAT Max'] = max(bwd_iat) if bwd_iat else 0
    features['Fwd Header Length'] = fwd_hdr_len
    features['Bwd Header Length'] = bwd_hdr_len
    features['Bwd IAT Min'] = min(bwd_iat) if bwd_iat else 0
    features['Bwd Packets/s'] = bwd_packet/flow_duration if flow_duration != 0 else 0
    features['Min Packet Length'] = min_packet_length
    features['FIN Flag Count'] = fin_flag_count
    features['RST Flag Count'] = rst_flag_count
    features['PSH Flag Count'] = psh_flag_count
    features['ACK Flag Count'] = ack_flag_count
    features['URG Flag Count'] = urg_flag_count
    features['Down/Up Ratio'] = down_up_ratio 
    features['Init_win_bytes_forward'] = init_win_bytes_fwd
    features['Init_win_bytes_backward'] = init_win_bytes_bwd
    return features




def sniff_packets(interface, duration=100):
    # Start capturing packets
    capture = pyshark.LiveCapture(interface=interface)
    start_time = datetime.now()
    for packet in capture.sniff_continuously(): 
        if 'TCP' in packet:
            dst_port = packet.tcp.dstport
            calculate_features(dst_port,packet)
        # Break the loop if duration exceeds
        if (datetime.now() - start_time).total_seconds() >= duration:
            break

           

if __name__ == "__main__":
    # Set the network interface to capture packets from
    interface = "eth0"  # Change this to your network interface name

    # Capture packets and extract features
    sniff_packets(interface)

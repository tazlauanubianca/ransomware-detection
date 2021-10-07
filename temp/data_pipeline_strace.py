from statistics import mean

import collections
import csv
import argparse
import os
import re
import json

PATH_LOGS = '{}/{}/logs/all.strace'
PATH_JSON = '{}/{}/reports/report.json'
linux_goodware = ["cheese", "eog", "evince", "firefox", "nautilus", "rhythmbox", "software-update-gtk", "thunderbird", "transmission-gtk", "ubuntu-software"]
linux_ransomware = ["bash-ransomware", "erebus", "hagz_all", "hagz"]

def network_info_stats(files_info, udps, tcps):
    print("Network Info Stats")
    
    dst_comm_list = []
    for file_name in files_info:
        dst_comm_list.extend(list(files_info[file_name]["dst_udp_traffic"]))
        dst_comm_list.extend(list(files_info[file_name]["dst_tcp_traffic"]))
    frequency = collections.Counter(dst_comm_list)
    
    print(frequency)
    print(mean(len(files_info[x]["dst_udp_traffic"]) + len(files_info[file_name]["dst_tcp_traffic"]) for x in files_info))
    
def write_results_to_file(files_info, system_calls, udps, tcps, deleted_files, changed_perms, read_files, written_files):
    header = ["file_name", "label", "test_sample"] + list(system_calls) + ["num_udp", "num_tcp"] + list(udps) + list(tcps) + \
            list("deleted_" + x for x in deleted_files) + \
            list("writte_" + x for x in written_files) + list("read_" + x for x in read_files) + \
            list("perms_" + x for x in changed_perms)
    
    # network_info_stats(files_info, udps, tcps)
    
    with open("results_all.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for file_info in files_info:
            data = [file_info]
            if "VirusShare" in file_info or file_info in linux_ransomware:
                data.append(1)
            else:
                data.append(0)
            
            if file_info in linux_ransomware or file_info in linux_goodware:
                data.append(1)
            else:
                data.append(0)
        
            for system_call in system_calls:
                if system_call in files_info[file_info]:
                    data.append(files_info[file_info][system_call])
                else:
                    data.append(0)

            data.append(files_info[file_info]["udp_traffic"])
            data.append(files_info[file_info]["tcp_traffic"])
            for udp in udps:
                if udp in files_info[file_info]["dst_udp_traffic"]:
                    data.append(1)
                else:
                    data.append(0)

            for tcp in tcps:
                if tcp in files_info[file_info]["dst_tcp_traffic"]:
                    data.append(1)
                else:
                    data.append(0)

            for deleted_file in deleted_files:
                if deleted_file in files_info[file_info]["deleted_files"]:
                    data.append(1)
                else:
                    data.append(0)
            
            for written_file in written_files:
                if written_file in files_info[file_info]["written_files"]:
                    data.append(1)
                else:
                    data.append(0)
            
            for read_file in read_files:
                if read_file in files_info[file_info]["read_files"]:
                    data.append(1)
                else:
                    data.append(0)
             
            for changed_file in changed_perms:
                if changed_file in files_info[file_info]["changed_perms"]:
                    data.append(1)
                else:
                    data.append(0)
            
            writer.writerow(data)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input', type=str, help='input file', required=True)
    parser.add_argument('-out', '--output', type=str, help='output file', required=True)
    args = parser.parse_args()

    input_dir = args.input
    folders = os.listdir(input_dir)
    system_calls = set()
    udps = set()
    tcps = set()
    created_files = set()
    deleted_files = set()
    changed_perms = set()
    unique_files = set()
    read_files = set()
    written_files = set()
    files_info = dict()

    for folder in folders:
        report_path = PATH_JSON.format(input_dir, folder)
        files_info[folder] = dict()
        files_info[folder]['deleted_files'] = []
        files_info[folder]['read_files'] = []
        files_info[folder]['written_files'] = []
        files_info[folder]['changed_perms'] = []
        
        with open(report_path, 'r') as f:
            try:
                data = json.load(f)
                #Network
                traffic_udp_data = set(x['dst'] for x in data['network']['udp'])
                traffic_tcp_data = set(x['dst'] for x in data['network']['tcp'])
                files_info[folder]['udp_traffic'] = len(traffic_udp_data)
                files_info[folder]['tcp_traffic'] = len(traffic_tcp_data)
                files_info[folder]['dst_udp_traffic'] = traffic_udp_data
                files_info[folder]['dst_tcp_traffic'] = traffic_tcp_data
                udps.update(files_info[folder]['dst_udp_traffic'])
                tcps.update(files_info[folder]['dst_tcp_traffic'])
            except:
                files_info[folder]['udp_traffic'] = 0
                files_info[folder]['tcp_traffic'] = 0
                files_info[folder]['dst_udp_traffic'] = []
                files_info[folder]['dst_tcp_traffic'] = []
       
        path = PATH_LOGS.format(input_dir, folder)
        with open(path, 'r') as f:
            for data in f:
                data = data.strip()
                parsed_data = list(filter(None, re.split('[\  : < > ( ) " ,]', data)))
                if "resumed" in parsed_data:
                    continue

                # Calls
                system_call = parsed_data[1]
                if '-' in system_call or '+' in system_call or '>' in system_call or '<' in system_call:
                    continue

                # Collect system call information
                system_calls.add(system_call)
                if system_call not in files_info[folder]:
                    files_info[folder][system_call] = 0
                files_info[folder][system_call] += 1
            
                # Collect information about operations on files
                if system_call == 'openat':
                    # Name of the file
                    file_name = parsed_data[3]
                    unique_files.add(file_name)
 
                    # Read/Write/Both
                    flags = parsed_data[2]
                    if 'O_RDONLY' in flags or 'O_RDWR' in flags:
                        read_files.add(file_name)
                        files_info[folder]['read_files'].append(file_name)

                    if 'O_WRONLY' in flags or 'O_RDWR' in flags:
                        written_files.add(file_name)
                        files_info[folder]['written_files'].append(file_name)

                if system_call == 'read':
                    file_name = parsed_data[3]
                    read_files.add(file_name)
                    files_info[folder]['read_files'].append(file_name)

                if system_call == 'write':
                    file_name = parsed_data[3]
                    written_files.add(file_name)
                    files_info[folder]['written_files'].append(file_name)
               
                if system_call == 'unlink':
                    file_name = parsed_data[2]
                    deleted_files.add(file_name)
                    files_info[folder]['deleted_files'].append(file_name)
                
                if system_call == 'chmod':
                    file_name = parsed_data[2]
                    changed_perms.add(file_name)
                    files_info[folder]['changed_perms'].append(file_name)
    
    write_results_to_file(files_info, system_calls, udps, tcps, deleted_files, changed_perms, read_files, written_files)

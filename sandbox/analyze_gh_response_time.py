import re
import numpy as np

def extract_response_times(log_file):
    response_times = []

    with open(log_file, 'r') as file:
        for line in file:
            match = re.search(r'\] took (\d+) ms', line)
            if match:
                response_times.append(float(match.group(1)))

    return response_times


def analyze_response_times(response_times):
    avg_response_time = np.mean(response_times)
    min_response_time = np.min(response_times)
    max_response_time = np.max(response_times)

    print(f'Average Response Time: {avg_response_time} ms')
    print(f'Minimum Response Time: {min_response_time} ms')
    print(f'Maximum Response Time: {max_response_time} ms')


if __name__ == '__main__':
    response_times = extract_response_times('sandbox/gh_copilot_log.txt')
    # Assuming response_times is the list of response times extracted from the log file
    analyze_response_times(response_times)



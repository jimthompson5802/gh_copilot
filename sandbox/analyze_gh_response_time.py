import re
import numpy as np
import pandas as pd

def extract_response_times(log_file):
    response_times = []

    with open(log_file, 'r') as file:
        for line in file:
            match = re.search(r'\] took (\d+) ms', line)
            if match:
                response_times.append(float(match.group(1)))

    return response_times


if __name__ == '__main__':
    response_times = extract_response_times('sandbox/gh_copilot_log.txt')
    
    # convert response times to pandas series
    response_times = pd.Series(response_times)

    # analyze response times
    print(f"Times msec:\n{response_times.describe()}")

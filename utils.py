import os
from datetime import datetime

# Generate experiment name
def generate_experiment_name(directory, base_name="experiment"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing_experiments = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    highest_index = 0
    for name in existing_experiments:
        if name.startswith(base_name) and len(name) > len(base_name):
            try:
                parts = name[len(base_name):].split('_')
                if len(parts) >= 2 and parts[0] == current_time:
                    index = int(parts[1])
                    highest_index = max(highest_index, index)
            except ValueError:
                continue
    next_index = highest_index + 1
    experiment_name = f"{base_name}_{current_time}_{next_index:03d}"
    return experiment_name
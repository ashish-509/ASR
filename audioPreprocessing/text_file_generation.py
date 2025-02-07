import os

def generate_text_files(dataset_path):
    label_mapping = {
        'A': 'sahara agadi',
        'B': 'sahara baya',
        'D': 'sahara daya',
        'P': 'sahara pachhadi',
        'R': 'sahara roka'
    }
    
    for subdir in ['train', 'validation']:
        subdir_path = os.path.join(dataset_path, subdir)
        
        if not os.path.exists(subdir_path):
            print(f"Skipping: {subdir_path} does not exist.")
            continue

        for filename in os.listdir(subdir_path):
            if filename.endswith('.wav'):
                label = label_mapping.get(filename[0], 'unknown') 
                
                text_filename = os.path.splitext(filename)[0] + '.txt'
                text_filepath = os.path.join(subdir_path, text_filename)
                
                with open(text_filepath, 'w', encoding='utf-8') as text_file:
                    text_file.write(label)
                
                print(f"Generated: {text_filepath} -> {label}")

dataset_path = '../dataset'  
generate_text_files(dataset_path)

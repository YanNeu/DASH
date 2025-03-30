import os
import json
import numpy as np

def load_object_image_folders(source_folder):
    object_image_folders = {}
    for object_name in sorted(os.listdir(source_folder)):
        if os.path.isdir(os.path.join(source_folder, object_name)):
            object_image_folders[object_name] = os.path.join(source_folder, object_name)
    
    return object_image_folders


def load_image_paths(image_folder):
    image_paths = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.png'):
            image_paths.append(os.path.join(image_folder, file_name))

    return image_paths


def remove_brackets(word):
    start, end = word.find('['), word.find(']')
    if ']' in word:
        return word[start+1:end]
    return word[start+1:]


def remove_angle_brackets(word):
    start, end = word.find('<'), word.find('>')
    if '>' in word:
        return word[start+1:end]
    return word[start+1:]


def remove_duplicate_objects(object_lists_raw):
    unique_lists = []
    for objects_raw in object_lists_raw:
        try:
            objects = remove_brackets(objects_raw)
            if ']' in objects_raw:
                objects = objects.split(',')
            else:
                objects = objects.split(',')[:-1]
            objects = [remove_angle_brackets(remove_brackets(obj.lower())).strip() for obj in objects]
            objects_unique = list(set(objects))

            if "" in objects_unique: objects_unique.remove("")
            
            if len(objects_unique) > 0:
                unique_lists.append(objects_unique)
        except:
            print(f'Skipping broken object list: {objects_raw}')
        
    return unique_lists
        

def merge_object_lists(object_lists):
    merged_objects = []
    for object_list in object_lists:
        merged_objects.extend(object_list)
    return list(set(merged_objects))


def save_object_lists(object_lists_dict, json_file):
    with open(json_file, 'w') as f:
        json.dump(object_lists_dict, f, indent=4)


def unique_lists(object_lists):
    object_tuples = [tuple(sorted(obj_list)) for obj_list in object_lists]
    unique_tuples = list(set(object_tuples)) 
    unique_lists = [list(obj_tuple) for obj_tuple in unique_tuples]
    return unique_lists


def load_object_lists(json_file):
    #json_file = os.path.join(load_dir, f'{object_name}_obj_lists.json')
    with open(json_file, 'r') as f:
            object_lists_dict = json.load(f)
    return object_lists_dict


def remove_empty(element_list):
    while "" in element_list: element_list.remove("")
    return element_list

def clean_element_list(element_list):
    cleaned_list = []
    for element in element_list:
        element = remove_brackets(element)
        element = remove_angle_brackets(element)
        if len(element) <= 1: continue
        cleaned_list.append(element)
    return cleaned_list

def merged_elements_frequencies(elements_lists, count_threshold=1):
    num_elements = len(elements_lists)
    merged_elements = []
    for element_list in elements_lists:
        element_list = clean_element_list(element_list)
        if len(element_list) == 0: continue
        merged_elements.extend(element_list)
    
    unique_elements, counts = np.unique(merged_elements, return_counts=True)
    sorted_indices = np.argsort(-counts)

    element_frequencies = {unique_elements[i]:counts[i]/num_elements for i in sorted_indices if counts[i] > count_threshold}

    return element_frequencies
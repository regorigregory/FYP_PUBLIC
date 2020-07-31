from pathlib import Path
import os
def get_project_dir()->Path:
    return Path(__file__).parent

def get_experiments_dir()->Path:
    return Path(os.path.join(get_project_dir(), "experiments"))

def get_benchmarking_dir()->Path:
    return Path(os.path.join(get_project_dir(), "benchmarking"))

def get_components_dir()->Path:
    return Path(os.path.join(get_project_dir(), "components"))

def get_datasets_dir()->Path:
    return Path(os.path.join(get_project_dir(), "datasets"))

def fix_win_rel_paths(path:str)->str:
    root = get_project_dir()
    cleaned_path = path.replace('\\', "/").replace("\t", "/t")
    #cleaned_path = path
    if ".." in cleaned_path:
        relative_index = cleaned_path.rfind("..")+3
        cleaned_path = cleaned_path[relative_index:]
        #print(cleaned_path)
    new_path =  root / Path(cleaned_path)
    #print(new_path)
    return new_path

if __name__ == "__main__":
    print(get_project_dir().is_dir())
    print(get_experiments_dir().is_dir())
    print(get_benchmarking_dir().is_dir())
    print(get_components_dir().is_dir())
from markitdown import MarkItDown
import os
from config_loader import load_config

config = load_config()
md = MarkItDown()
PERMANENT_MD_PATH = config['STORAGE_PATH'] + "/md"
os.makedirs(PERMANENT_MD_PATH, exist_ok=True)

def convert_file_to_markdown(file_path:str,is_permanent:bool=False):
    global PERMANENT_MD_PATH
    try:
        result = md.convert(file_path)
        if is_permanent:
            filename = os.path.basename(file_path).split('.')[0] + ".md"
            file_path = os.path.join(PERMANENT_MD_PATH, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(result.text_content)
        return result.text_content
    except Exception as e:
        raise Exception(f"Error converting file: {str(e)}")
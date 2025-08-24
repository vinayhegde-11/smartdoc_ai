from markitdown import MarkItDown
import os

PERMANENT_MD_PATH = ""
TEMP_MD_PATH = ""

def convert_file_to_markdown(file_path,is_permanent:bool=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    md = MarkItDown()
    try:
        result = md.convert(file_path)
        return result.text_content
    except Exception as e:
        raise Exception(f"Error converting file: {str(e)}")
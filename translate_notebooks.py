import json
import glob
import re
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='en', target='fr')

def translate_text(text):
    try:
        # Avoid translating empty or whitespace only strings
        if not text.strip(): return text
        return translator.translate(text)
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return text

notebooks = glob.glob('f:/GreenWings/Computer-Vision-Model-For-Plant-Disease-Identification-main/*.ipynb')

for nb_path in notebooks:
    print(f"Processing {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)
        
    changes_made = False
    for cell in nb_data.get('cells', []):
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell.get('source', []):
                match = re.search(r'^(\s*)#\s*(.+)$', line)
                if match:
                    spaces = match.group(1)
                    comment_text = match.group(2)
                    
                    ends_with_newline = comment_text.endswith('\n')
                    if ends_with_newline:
                        comment_text = comment_text[:-1]
                        
                    translated_comment = translate_text(comment_text)
                    
                    new_line = f"{spaces}# {translated_comment}"
                    if ends_with_newline:
                        new_line += '\n'
                    new_source.append(new_line)
                    changes_made = True
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
        elif cell['cell_type'] == 'markdown':
            new_source = []
            source = cell.get('source', [])
            if isinstance(source, list):
                for line in source:
                    if line.strip():
                        has_nl = line.endswith('\n')
                        text_to_translate = line[:-1] if has_nl else line
                        translated = translate_text(text_to_translate)
                        new_source.append(translated + ('\n' if has_nl else ''))
                        changes_made = True
                    else:
                        new_source.append(line)
            cell['source'] = new_source

    if changes_made:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb_data, f, indent=1, ensure_ascii=False)
        print(f"Updated {nb_path}")
print("Done!")

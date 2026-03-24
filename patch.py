import glob
import py_compile

for f in glob.glob('*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        c = file.read()
    c = c.replace('from keras.applications.mobilenet \nimport', 'from keras.applications.mobilenet import')
    c = c.replace('from zipfile \nimport', 'from zipfile import')
    c = c.replace('axi\ns=-1)', 'axis=-1)')
    c = c.replace('axi\naxis=-1)', 'axis=-1)')
    c = c.replace('s = \n-1)', 'axis=-1)')
    c = c.replace('classe\ns=', 'classes=')
    c = c.replace('s= \n', 's=')
    c = c.replace('s = \n', 's=')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(c)

has_error = False
for f in glob.glob('*.py'):
    if f in ['translate_notebooks.py', 'fetch_ui.py', 'convert_nb.py', 'patch.py', 'patch_and_push.py']: continue
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        print("Error in:", f)
        print(e)
        has_error = True

if not has_error:
    print("ALL COMPILED OK")

import glob
import os
import py_compile

for f in glob.glob('*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        c = file.read()
    c = c.replace('from keras.preprocessing.image \nimport', 'from keras.preprocessing.image import')
    c = c.replace('from rembg \nimport', 'from rembg import')
    c = c.replace('pat\nh=', 'path=')
    with open(f, 'w', encoding='utf-8') as file:
        file.write(c)

has_error = False
for f in glob.glob('*.py'):
    if f in ['translate_notebooks.py', 'fetch_ui.py', 'convert_nb.py', 'patch_and_push.py']: continue
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        print("Still error in:", f)
        print(e)
        has_error = True

if not has_error:
    print("No errors! Pushing to git...")
    os.system("git add .")
    # Commit message
    os.system('git commit -m "Convert notebooks and build monitoring UI"')
    os.system("git push origin main")
    print("ALL DONE")
else:
    print("There are still errors.")

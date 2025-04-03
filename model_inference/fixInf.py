from pathlib import Path
for i in range(0, 500):
    try:
        inFile = Path('ppl' + f"/{i}.txt").resolve()
        inF = open(inFile, 'r', encoding='utf-8')
        response = inF.read()
        inF.close()
        print(response)
        if response == 'inf':
            print("Change")
            outFile = open(inFile, 'w', encoding='utf-8')
            outFile.write(str(0))
            outFile.close()
    except:
        pass
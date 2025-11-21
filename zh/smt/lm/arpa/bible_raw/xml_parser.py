import sys
import xml.etree.ElementTree as ET

if sys.argv[2] == 'cn':
    from langconv.converter import LanguageConverter
    from langconv.language.zh import zh_cn

    lc_cn = LanguageConverter.from_language(zh_cn)  # target variant set to zh-cn
    write_file = sys.argv[3]
else:
    write_file = sys.argv[2]

tree = ET.parse(sys.argv[1])
root = tree.getroot()

sents = []
for seg in root.iter('seg'):
    t = seg.text
    if sys.argv[2] == 'cn':
        sents.append(lc_cn.convert(t.strip()))
    else:
        sents.append(t.strip())
with open(write_file, 'w') as f:
    for sent in sents:
        f.write(sent + '\n')    


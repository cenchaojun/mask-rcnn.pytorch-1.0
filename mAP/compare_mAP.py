from xlwt import *
import pdb

classes = ['肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '主动脉结增宽', '膈面异常', '结节',
           '肿块', '异物', '气胸', '肺气肿', '骨折', '钙化', '乳头影', '弥漫性结节', '肺不张',
           '多发结节', '心影增大', '脊柱侧弯', '纵隔变宽', '肺门增浓', '膈下游离气体', '肋骨异常',
           '肺结核', '皮下气肿', '主动脉钙化', '空洞', '液气胸', '肋骨缺失', '肩关节异常']

map_file1 = 'results/results_original.txt'
map_file2 = 'results/results_four.txt'

book = Workbook(encoding='utf-8')
sheet = book.add_sheet('mAP')
sheet.write(0, 1, label='mAP')

with open(map_file1) as f1:
    content = f1.readlines()
    sheet.write(1,0,label='original')
    j = 1
    for line in content:
        line = line.strip()
        for i in range(len(classes)):
            if ('= ' + classes[i] + ' AP') in line:
                j += 1
                mAP = line.split(' ')[0].split('%')[0]
                sheet.write(0,j,label=classes[i])
                sheet.write(1,j,label=mAP)
                break
        if 'mAP = ' in line:
            mAP = line.split(' ')[-1].split('%')[0]
            sheet.write(1, 1, label=mAP)

with open(map_file2) as f2:
    content = f2.readlines()
    sheet.write(2,0,label='new')
    j = 1
    for line in content:
        line = line.strip()
        for i in range(len(classes)):
            if ('= ' + classes[i] + ' AP') in line:
                j += 1
                mAP = line.split(' ')[0].split('%')[0]
                sheet.write(2,j,label=mAP)
                break
        if 'mAP = ' in line:
            mAP = line.split(' ')[-1].split('%')[0]
            sheet.write(2, 1, label=mAP)

book.save('results/mAP_temp.xls')

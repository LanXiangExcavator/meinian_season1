# -*- coding: utf-8 -*-
"""
# @Time    :2018/5/7 下午9:13
# @Author  :Xuxian
"""

import numpy as np
import pandas as pd
import re


# 拼接数据
def stitching(part_1, part_2):
    data = pd.concat([part_1, part_2])
    # 多次检查项目进行拼接
    vid_table = data.groupby(['vid', 'table_id']).size().reset_index()
    vid_table['new_index'] = vid_table['vid'] + '_' + vid_table['table_id']
    vid_table_dup = vid_table[vid_table[0] > 1]['new_index']
    data['new_index'] = data['vid'] + '_' + data['table_id']
    dup_part = data[data['new_index'].isin(list(vid_table_dup))]
    dup_part = dup_part.sort_values(['vid', 'table_id'])
    unique_part = data[~data['new_index'].isin(list(vid_table_dup))]

    # 重复数据的拼接操作
    def merge_table(df):
        df['field_results'] = df['field_results'].astype(str)
        if df.shape[0] > 1:
            merge_df = " ".join(list(df['field_results']))
        else:
            merge_df = df['field_results'].values[0]
        return merge_df

    data_dup = dup_part.groupby(['vid', 'table_id']).apply(merge_table).reset_index()
    data_dup.rename(columns={0: 'field_results'}, inplace=True)
    data_res = pd.concat([data_dup, unique_part[['vid', 'table_id', 'field_results']]])
    data = data_res.pivot(index='vid', columns='table_id')['field_results'].reset_index()
    data.to_csv('../data/data_all.csv', index=False)


# part2数值型数据拼接
def part2_pivot(part_2):
    part_2 = part_2.drop_duplicates(['vid', 'table_id'], keep='last')
    data2 = part_2.pivot(index='vid', columns='table_id')['field_results'].reset_index()
    data2.to_csv('../data/data2.csv', index=False)


# 数值型特征
def numeric_feature(part_2):
    data2 = pd.read_csv('../data/data2.csv')

    def strQ2B(ustring):
        """把字符串全角转半角"""
        # ustring = ustring.decode('utf8')
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            elif (inside_code >= 0xFF01) and (inside_code <= 0xFF5E):
                inside_code -= 0xfee0
            rstring += chr(inside_code)
        return rstring

    # 提取数值
    def find_nums(x):
        if x == x:

            str_num = str(x)
            pat = r"\d+\.?\d*$"
            tmp = re.findall(pat, str_num)
            tmp = [float(strQ2B(x)) for x in tmp]
            if len(tmp) == 0:
                return -1
            return sum(tmp) / len(tmp)
        else:
            return x

    # 字典
    map2 = {
        '详见报告单': np.nan, '详见报告': np.nan, '详见纸质报告': np.nan, '未做': np.nan, '见图文报告': np.nan, '标本已退检': np.nan,
        '见TCT': np.nan, '***.**': np.nan, '**.*': np.nan, '详见报告。': np.nan, '详见检验单': np.nan, '见报告单': np.nan,
        '见报告': np.nan,
        '见刮片': np.nan, '结果见TCT': np.nan, '经复查': np.nan, '详见纸质报告单': np.nan, '见纸质报告单': np.nan, '未查': np.nan,
        '详见化验单': np.nan, '详见图文报告': np.nan, '样本经复查': np.nan, 'HIV感染待确定': np.nan, '.': np.nan,
        '-----': np.nan, 'Ⅱv': np.nan, '其他': np.nan, '/': np.nan, '基因缺失': np.nan,
        '送检玻片一张，镜检：上皮细胞量较少，建议复查。': np.nan,

        'O': 1, 'O型': 1, '“O”型': 1, ' O  型': 1, 'O   RH阳性': 1, 'O  RH阳性': 1, 'O 型': 1, '“O”': 1, '“O”型': 1, 'O  型': 1,
        'O型  Rh阳性': 1,
        'O型,RH(+)': 1, 'O型血': 1, 'Rh阳性O型': 1,
        'A': 2, 'A型': 2, '“A”型': 2, 'A   RH阳性': 2, 'A型，RH(+)': 2, '  A 型': 2, ' A  型': 2, 'Rh阳性A型': 2, '(A)': 2,
        '“A”': 2,
        '“A”型': 2,
        'A  RH阳性': 2, 'A  型': 2, 'A 阳性': 2, 'a型': 2,
        'B': 3, 'B型': 3, '“B”型': 3, 'Rh阳性B型': 3, ' B  型': 3, 'B   RH阳性': 3, 'B  型': 3, 'B 型': 3, 'B型,RH(+)': 3,
        'B型血': 3,
        '“B”': 3,
        'B  RH阳性': 3, 'B 型 RH 阳性': 3, 'b型': 3, 'B型+Rh阳性': 3,
        'AB': 4, 'AB型': 4, '“AB”型': 4, 'AB 型': 4, ' AB  型': 4, '(AB)': 4, '“AB”': 4, 'AB   RH阳性': 4, 'ab型': 4,

        '-': 0, '阴性': 0, '阴性（-）': 0, '阴性(-)': 0, '阴性 -': 0, '--': 0, '----': 0, '=- 阴性': 0, '---': 0, ' 阴性': 0,
        '- 阴性': 0,
        '+-': 1, '阳性(低水平)': 1, '弱阳': 1, '弱阳性(±)': 1, '弱阳性': 1, '极弱阳': 1, '微弱阳性': 1,
        '+': 2, '阳性(+)': 2, '+/HP': 2, '阳性（+）': 2, '＋': 2, ' +': 2, '陽性': 2, '﹢': 2, '阳性+': 2, '阳性(轻度)': 2, '阳性': 2,
        '++': 3, '++/HP': 3, '＋＋': 3, ' ++': 3, '阳性(中度)': 3,
        '+++': 4, '+++/HP': 4, '＋＋＋': 4, '++++/HP': 4, '阳性(重度)': 4,
        '++++': 5,

        'Ⅰ': 1, 'I': 1, 'Ⅰ°': 1, 'Ⅰ度': 1, 'i°': 1,
        'Ⅱ': 2, 'Ⅱ度': 2, 'II': 2, 'ii°': 2, 'Ⅱ°': 2,
        'Ⅲ': 3, 'Ⅲ度': 3, 'III': 3, 'iii°': 3, 'Ⅲ°': 3,
        'Ⅳ': 4, 'Ⅳ度': 4, 'Ⅳ°': 4, 'iv°': 4,

        '混浊': 1, '浑浊': 1, '白浊': 1,
        '透明': 0, '无色': 0, '见透明': 0, '微混': 0, '微浑': 0,

        '淡黄色': 2, '浅黄色': 2,
        '黄色': 3, 'yellow': 3, '深黄色': 3,
        '黄褐色': 4, '黄棕色': 4,
        '褐色': 5,
        '暗红色': 6, '淡红色': 6, '红色': 6,
        '黑色': 8,

        '未见': 0, '正常': 0, '未检出': 0, 'Normal': 0, '无敏感菌': 0, '无': 0, '未见异常': 0, 'NormaL': 0,
        '未见上皮内病变或恶性病变：萎缩性细胞改变伴炎症，建议定期(一~三)年后复查。': 0, '未提示': 0,
        '未见上皮内病变或恶性病变，萎缩性细胞改变伴炎症，建议定期(一~三)年后复查。': 0, '未生长': 0,
        '未见上皮内病变或恶性病变：反应性细胞改变，建议定期一年后复查。': 0,
        '未见上皮内病变或恶性病变，炎性反应性细胞改变，建议定期一年后复查。': 0,
        '未见上皮内病变或恶性病变，建议定期一年后复查。': 0,
        '(细胞量稍少)未见上皮内病变或恶性病变，炎性反应性细胞改变，建议定期一年后复查。': 0,
        '（细胞量稍少)未见上皮内病变或恶性病变，炎性反应性细胞改变，建议根据临床情况随诊或定期半年后复查。': 0,
        '(细胞量稍少)未见上皮内病变或恶性病变：萎缩性细胞改变伴炎症，建议定期(一~三)年后复查。': 0,

        '送检玻片一张，镜检：正常范围内，未见癌细胞。': 0, '未检测到缺失': 0,
        '未检测到缺失。': 0, '待复查': 0,
        ' 未见': 0,

        '可疑': 1, '少许': 1, '少见': 1, '少量': 1, '偶见': 1, '少': 1, '少数': 1, '偶 见': 1,
        '查见': 2, '异常': 2, '检出': 2, '大量': 2, '多见': 2, '中量': 2, '检到': 2,
        '重度': 3, '满视野': 3, '散在满视野': 3,

        'S': 1, '敏感(S)': 1, '敏感': 1,
        '中敏': 2, '中度': 2, '中度敏感(MS)': 2,
        '耐药': 3, 'R': 3, '耐药(R)': 3,

        '稀': 1, '软': 1, '软便': 1, '软,糊状': 1, '半稀便': 1,
        '硬': 2,

        '草酸钙结晶': 1, '草酸钙结晶少量': 1, '上皮细胞': 1, '上皮细胞少量': 1, '可见粘液丝': 1,
        '颗粒管型偶见': 1, '见黏液丝': 1, '扁平上皮细胞': 1,
        '磷酸盐结晶+': 2, '黏液丝+': 2, '尿酸结晶+': 2, '尿酸盐结晶+': 2, '透明管型+': 2, '查见透明管型': 2,
        '粘液丝+': 2, '细菌+': 2, '草酸钙结晶+': 2, '上皮细胞+': 2,
        '草酸钙结晶++': 3, '酵母样细胞++': 3, '上皮细胞++': 3, '粘液丝++': 3, '粘液丝+++': 3,

        '巴氏I级,未见异常': 1,
        '未检测到突变。': 0,

        'HIV抗体阴性(-)': 0, 'HIV抗体阴性': 0,

        '非典型鳞状细胞，不除外高度病变(ASC-H),建议阴道镜检查并活检。': 4,
        '非典型鳞状细胞，不除外高度病变（asc-h），建议阴道镜检查并宫颈活检。': 4,
        '鳞状上皮内高度病变(HSIL)，建议活检。': 4,
        '非典型腺细胞(AGUS),不能明确意义，建议进一步检查。': 4,
        '查见不能明确意义的非典型鳞状上皮细胞（ASCUS），建议做HPV相关检查或阴道镜检。': 4,
        '非典型鳞状细胞(ASCUS),建议检测HPV病毒或定期(三~六)月复查。': 2,
        '非典型鳞状细胞(ascus)，建议检测hpv病毒或定期(三～六月)复查。': 2,
        '非典型鳞状细胞(ASCUS),建议进一步检查或定期(三~六)月复查。': 2,
        '鳞状上皮内低度病变(LSIL),建议阴道镜检查。': 4,

        '未见上皮内病变或恶性病变（NILM），建议临床定期复查。': 1,
        '非典型鳞状细胞(ASCUS),建议阴道镜检查。': 1,
        '(细胞量稍少)未见上皮内病变或恶性病变，炎性反应性细胞改变，建议根据临床情况随诊或定期半年后复查。': 1,
        '未见上皮内病变或恶性病变（NILM），鳞状上皮炎症反应性改变，建议临床定期复查。': 1,
        '未见上皮内病变或恶性病变：萎缩性细胞改变伴炎症，可见真菌，形态符合念珠菌，建议治疗后复查。': 1,
        '未见上皮内病变或恶性病变：反应性细胞改变，可见真菌，形态符合念珠菌，建议治疗后复查。': 1,
        '未见上皮内病变或恶性病变，炎性反应性细胞改变，可见滴虫，建议治疗后复查。': 1,
        '未见上皮内病变或恶性病变（NILM），鳞状上皮炎症反应性改变，见真菌，形态符合念珠菌属。': 1,
        '未见上皮内病变或恶性病变（NILM），鳞状上皮炎症反应性改变，见线索细胞，提示细菌性阴道病，请结合临床考虑。': 1,
        '未见上皮内病变或恶性病变（NILM），鳞状上皮萎缩、炎症反应性改变，建议临床定期复查。': 1,
        '未见上皮内病变或恶性病变（NILM），鳞状上皮萎缩性改变，建议临床定期复查。': 1,
        '未见上皮内病变或恶性病变，炎性反应性细胞改变，可见真菌，形态符合念珠菌，建议治疗后复查。': 1,
        '未见上皮内病变或恶性病变：萎缩性细胞改变伴炎症，建议定期（一～三）年后复查。': 1,
        '滴虫偶见': 1,
        '检出滴虫': 1,

        '细胞量稍少)未见上皮内病变或恶性病变，萎缩性细胞改变伴炎症，建议定期(一~三)年后复查。': 0,
        '未见上皮内病变': 0,

        '无解脲脲原体和人型支原体生长': 0,

        '降脂后复查': np.nan, '血液': np.nan, '中度脂血': np.nan, '女性肿瘤指标': np.nan, '混合性红细胞': np.nan, '宫颈刮片：': np.nan,
        '中介': np.nan, '非均一性': np.nan
    }

    part2_table_id = part_2['table_id'].value_counts().reset_index()
    part2_table_id.columns = ['table_id', 'I']

    # 提取出现次数大于 400的数值特征
    index_list = part2_table_id[part2_table_id['I'] > 400]['table_id'].tolist()

    def mapping(x):
        try:
            return map2[x]
        except:
            return x

    for i in index_list:
        data2[i + '_new'] = data2[i].apply(mapping)
        data2[i + '_new'] = data2[i + '_new'].apply(find_nums)

    dict_100010 = {'-': 0, '阴性': 0, '0(-)': 0, '-     10CELL/uL': 0, '+-     10CELL/uL': 0, '未见': 0,
                   '-    10CELL/uL': 0,
                   '-       0CELL/uL': 0,
                   '未做': np.nan, '透明': np.nan,
                   '+-': 1, '10(+-)': 1, '+-       0CELL/uL': 1, '+-   200CELL/uL': 1,
                   '+': 2, '阳性(+)': 2, '阳性(1+)': 2, '1+': 2, '25(+1)': 2, '+1': 2, '１＋': 2, '+1     25CELL/uL': 2,
                   '++': 3, '2+': 3, '80(+2)': 3, '２＋': 3,
                   '+++': 4, '200(+3)': 4, '3+': 4, '3': 4, }
    data2['100010_new'] = data2['100010'].apply(lambda x: dict_100010[x] if x == x else x)

    dict_360 = {'-': 0, '0': 0, '阴性': 0,
                '+-': 1,
                '+': 2, '阳性(+)': 2, '++': 2, '+++': 2, '1+': 2, '+1': 2, '3+': 2, '2+': 2}

    def f_360(x):
        if x == x:
            try:
                return dict_360[x]
            except:
                np.nan
        else:
            x

    data2['360_new'] = data2['360'].apply(f_360)

    index_list_new = [x + '_new' for x in index_list]
    index_list_new.append('vid')
    return data2[index_list_new]


# 文本特征
def process_text(data):
    dict_0406 = {'未见异常': 0, '未触及': 0, '肋下未触及': 0, '未见异常 未见异常': 0, '肝肿大': 1, '未触及 未触及': 0, '不大': 0, '未及': 0,
                 '可触及右肋下1.0cm 剑突下4.0cm': 1, '可触及右肋下1.0cm 剑突下3.0cm': 1, '可触及左肋下1cm 剑突下1cm': 1,
                 '可触及右肋下2.0cm 剑突下5.0cm': 1, '可触及右肋下1.5cm 剑突下2cm': 1, '可触及右肋下2.0cm 剑突下4.0cm': 1,
                 '可触及右肋下3.0cm 剑突下6.0cm': 1, '可触及右肋下1cm 剑突下1.5cm': 1, '可触及右肋下2cm 剑突下2cm': 1,
                 '可触及右肋下1.5cm 剑突下2.5cm': 1, '肋下约1cm': 1, '腹壁厚触诊不满意': 2, '可触及右肋下1.5cm 剑突下4.0cm': 1,
                 '有移动性浊音': 1, '可触及右肋下0.5cm 剑突下3.0cm': 1, '肝肿大, 可触及右肋下7cm 剑突下5cm': 1,
                 '可触及右肋下1.0cm 剑突下3.5cm': 1, '正常': 0}

    # 0406:肝脏触诊  0:37866 正常 | 1:46 异常 | -1:19386 空
    data['Medical_examination_liver'] = data['0406'].apply(
        lambda x: dict_0406[str(x).strip()] if str(x).strip() in dict_0406.keys() else -1)

    dict_0407 = {'未见异常': 0, '未触及': 0, '未见异常 未见异常': 0, '脾脏切除': 1, '可触及': 1, '未触及 未触及': 1, '脾肿大': 1, '不大': 0, '未及': 0,
                 '弃查': -1, '未 触及': 0, '脾脏可触及肋下1.5cm': 1, '脾脏有压痛': 1, '脾脏可触及肋下2.5cm': 1, '腹壁厚触诊不满意': 2}
    # 0407:脾脏触诊  -1:16014|空 0:41245|正常 1:38|异常 2:1|腹壁厚触诊不满意
    data['Medical_examination_spleen'] = data['0407'].apply(
        lambda x: dict_0407[str(x).strip()] if str(x).strip() in dict_0407.keys() else -1)

    def logic_judge(x):
        if x == True:
            return 1
        elif x == False:
            return 0
        else:
            return -1

    # 0409内科检查既往病史
    # hypertension高血压
    data['Medical_history_hypertension'] = data['0409'].str.contains(
        '血压', na=-1).apply(logic_judge)
    # Hyperlipidemia高血脂
    data['Medical_history_hyperlipidemia'] = data['0409'].str.contains(
        '脂', na=-1).apply(logic_judge)
    # hyperglycemia高血糖or糖尿病
    data['Medical_history_hyperglycemia'] = data['0409'].str.contains(
        '糖', na=-1).apply(logic_judge)
    # coronary冠心病 292
    data['Medical_history_coronary'] = data['0409'].str.contains(
        '冠心', na=-1).apply(logic_judge)
    # Arrhythmia 心律不齐 1366
    data['Medical_history_arrhythmia'] = data['0409'].str.contains(
        '心律', na=-1).apply(logic_judge)

    def logic_0413(x):
        # np.nan==np.nan return False
        list_0413 = ['未见异常', '无', '未见异常 未见异常', '未见明显异常', '未见明显异常 未见明显异常']
        if x == x:
            if (x == '未查'):
                return -1
            elif x in list_0413:
                return 0
            else:
                return 1
        else:
            return -1

    # 内科综合
    data['Medical_examination_synthesize'] = data['0413'].apply(
        lambda x: logic_0413(x))

    def logic_0420(x):
        # np.nan==np.nan return False
        list_0420 = ['未见异常', '正常', '未闻及异常', '正常 正常']
        if x == x:
            if x in list_0420:
                return 0
            else:
                return 1
        else:
            return -1

    # 内科心音
    data['Medical_examination_heart_sound'] = data['0420'].apply(
        lambda x: logic_0420(x))

    # 心律不齐
    # 窦性心律不齐(窦性心律) 114
    def logic_0421(x):
        list_0421 = ['整齐', '齐', '整齐 整齐', '整齐.心律过缓', '=整齐', '整齐心率稍慢', '+666666666666666666666666666666666666666666666整齐',
                     '整齐66', '整齐.心动过速', '0整齐', '整齐.心律过速', '++-------------整齐', '齐，心动过速']
        if x == x:
            if x in list_0421:
                return 0
            else:
                return 1
        else:
            return -1

    # 心律不齐
    data['Medical_examination_arrhythmia'] = data['0421'].apply(
        lambda x: logic_0421(x))

    dict_0422 = {'未见异常': 0, '无神经定位体征': 1,
                 '生理反射存在，病理反射未引出': 2, '无神经定位体征 无神经定位体征': 1}
    # 内科神经
    # -1:43224 | 0:10317 | 1:2202 | 2:1555
    data['Medical_examination_nervosum'] = data['0422'].apply(
        lambda x: dict_0422[str(x).strip()] if str(x).strip() in dict_0422.keys() else -1)

    ###########
    def logic_0423(x):
        bad = r"粗|减弱|哮鸣|消失"
        normal = r"清|常"
        if x == x:
            if re.search(bad, str(x)):
                return 1
            elif re.search(normal, str(x)):
                return 0
            else:
                return -1
        else:
            return -1

    # 肺呼吸音
    # 0:40359 | -1:16797 | 1:142
    data['Medical_examination_respiratory_sound2'] = data['0423'].apply(
        lambda x: logic_0423(x))

    def strQ2B(ustring):
        """把字符串全角转半角"""
        # ustring = ustring.decode('utf8')
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            elif (inside_code >= 0xFF01) and (inside_code <= 0xFF5E):
                inside_code -= 0xfee0
            rstring += chr(inside_code)
        return rstring

    # 提取数值
    def find_num(x):
        str_num = str(x).strip()
        pat = r"\d{1,3}"
        tmp = re.findall(pat, str_num)
        tmp = [int(strQ2B(x)) for x in tmp]
        return sum(tmp) / len(tmp)

    # data['0424'].apply()

    def logic_0424(x):
        if x == x:
            try:
                return find_num(x)
            except:
                if x == '未见异常':
                    return 72
                elif x == '窦性心动过速':
                    return 130
                elif x == '心动过缓':
                    return 50
        else:
            return -1

    # 心率
    data['Medical_examination_heart_rate'] = data['0424'].apply(logic_0424)

    def logic_0426(x):
        bad = r"Ⅲ|Ⅳ|3|III"
        normal = r"未|常|无"
        if x == x:
            if re.search(bad, str(x)):
                return 2
            elif re.search(normal, str(x)):
                return 0
            else:
                return 1
        else:
            return -1

    # 心音杂音
    data['Medical_examination_heart_voice2'] = data['0426'].apply(logic_0426)

    # 肝胆区按压
    dict_0431 = {'未见异常': 0, '无': 0, '无压痛点': 0, '叩击痛': 1, '未见异常 未见异常': 0, '压痛': 1, '无 无': 0, '未触及': -1, '压痛, 叩击痛': 1,
                 '胆囊切除术后': -1, '叩击痛, 叩击痛': 1}
    data['Medical_examination_liver_gall'] = data['0431'].apply(
        lambda x: dict_0431[str(x).strip()] if str(x).strip() in dict_0431.keys() else -1)

    # hypertension高血压 4729
    data['Medical_history_hypertension_1'] = data['0434'].str.contains(
        '血压', na=-1).apply(logic_judge)
    # Hyperlipidemia高血脂 1691
    data['Medical_history_hyperlipidemia_1'] = data['0434'].str.contains(
        '脂', na=-1).apply(logic_judge)
    # hyperglycemia高血糖or糖尿病 1623
    data['Medical_history_hyperglycemia_1'] = data['0434'].str.contains(
        '糖', na=-1).apply(logic_judge)
    # coronary冠心病 404
    data['Medical_history_coronary_1'] = data['0434'].str.contains(
        '冠', na=-1).apply(logic_judge)
    # Thyroid 甲状腺功能减退
    data['Medical_history_thyroid_1'] = data['0434'].str.contains(
        '甲状腺功能减退', na=-1).apply(logic_judge)
    # Pregnant 怀孕
    data['Medical_history_pregnant_1'] = data['0434'].str.contains(
        '剖宫产术后|孕', na=-1).apply(logic_judge)

    def find_zhifanggan(data):
        #   无数据，无脂肪肝：0
        #  轻、中、重：1,2,3
        if data == data:
            data = str(data).replace(' ', '')
            reg = r'脂肪肝(（\S+?）)'
            lev = r'轻,中,重'.split(',')
            try:
                result = re.search(reg, data).group(1)
            except:
                result = None
                return -1

            if not result:
                return 0
            else:
                for index in range(len(lev)):
                    if lev[index] in result:
                        content = re.search(reg, data).group(0)
                        return index + 1
                return 1  # 脂肪肝后没有等级，默认为轻度脂肪肝
        else:
            return -1

    # 脂肪肝
    data['doppler_fatty_liver'] = data['0102'].apply(find_zhifanggan)

    def logic_0911(x):
        if x == x:
            if re.search(r"淋巴结", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['LinBaJie'] = data['0911'].apply(logic_0911)

    def logic_0929_jiejie(x):
        if x == x:
            if re.search(r"结节", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['RuxianJiejie'] = data['0929'].apply(logic_0929_jiejie)

    def logic_0929_zengsheng(x):
        if x == x:
            if re.search(r"增生", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['RuxianZengsheng'] = data['0929'].apply(logic_0929_zengsheng)

    def logic_0949(x):
        if x == x:
            if re.search(r"静脉曲张", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['JingmaiQuZhang'] = data['0949'].apply(logic_0949)

    def logic_0972(x):
        if x == x:
            if re.search(r"痔", str(x)):
                return 1
            elif re.search(r"弃查|自述不查|未检", str(x)):
                return -1
            else:
                return 0
        else:
            return -1

    # 痔疮
    data['zhi'] = data['0972'].apply(logic_0972)

    def logic_0975(x):
        if x == x:
            if re.search(r"脂肪瘤", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    # 脂肪瘤
    data['ZhiFangLiu'] = data['0975'].apply(logic_0975)

    def logic_0984(x):
        if x == x:
            if re.search(r"前列腺|增|硬", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    # 前列腺
    data['QianLieXianZengSheng'] = data['0984'].apply(logic_0984)

    # 动脉硬化与斑块
    def logic_0102_bankuai(x):
        if x == x:
            if re.search(r"硬化|斑块", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['BanKuai'] = data['0102'].apply(logic_0102_bankuai)

    # 左心室舒张功能减低
    def logic_0102_Zuoxinfang(x):
        if x == x:
            if re.search(r"左心室舒张功能减低", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['ZuoxinfangShuZhang'] = data['0102'].apply(logic_0102_Zuoxinfang)

    # 动脉反流
    def logic_0102_fanliu(x):
        if x == x:
            if re.search(r"反流", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['FanLiu'] = data['0102'].apply(logic_0102_fanliu)

    # 左室高电压
    def logic_1001(x):
        if x == x:
            if re.search(r"高电压", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['gaodianya'] = data['1001'].apply(logic_1001)

    # 角膜老年环
    def logic_1305(x):
        if x == x:
            if re.search(r"老年环", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['JiaoMOLaoNianHuan'] = data['1305'].apply(logic_1305)

    # 视网膜动脉硬化
    def logic_1316(x):
        if x == x:
            if re.search(r"动脉", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['ShiWangMoDongMaiYingHua'] = data['1316'].apply(logic_1316)

    # 脑动脉
    def logic_1402(x):
        if x == x:
            if re.search(r"减慢|降低|硬化", str(x)):
                return 1
            elif re.search(r"增快", str(x)):
                return 2
            else:
                return 0
        else:
            return -1

    data['NaoDongMai'] = data['1402'].apply(logic_1402)

    # 动脉血管弹性
    def logic_4001(x):
        if x == x:
            if re.search(r"减弱趋势", str(x)):
                return 1
            elif re.search(r"轻度|临界", str(x)):
                return 2
            elif re.search(r"中度", str(x)):
                return 3
            elif re.search(r"重度", str(x)):
                return 4
            elif x in ['详见纸质报告', '详见图文报告单', '结论详见报告单。', '提示因血管堵塞而本次动脉硬化检查不能评价血管硬度',
                       '提示因血管堵塞而本次动脉硬化检查不能评价血管硬度, 左右下肢血管堵塞可能',
                       '因双下肢静脉曲张本次动脉硬化检查不能评价血管硬度', '详见纸质报告, 详见纸质报告']:
                return -1
            elif x in ['血管弹性度正常，血管腔未见狭窄', '血管弹性良好', '血管弹性良好, 下肢血管未见异常', '血管弹性度正常。',
                       '血管弹性正常', '下肢血管未见异常, 血管弹性良好', '血管弹性良好, 详见纸质报告', '血管弹性度正常',
                       '血管弹性度正常，血管腔未见狭窄详见纸质报告']:
                return 0
            elif re.search(r"可能", str(x)):
                return 1
            else:
                return 2
        else:
            return -1

    data['DongMaiYingHua'] = data['4001'].apply(logic_4001)

    # 肝囊肿
    def logic_0102_gannangzhong(x):
        if x == x:
            if re.search(r"肝囊肿", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['GanNangZhong'] = data['0102'].apply(logic_0102_gannangzhong)

    # 脾脏增大
    def logic_0102_pizang(x):
        if x == x:
            if re.search(r"脾脏增", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['PiZangZengDa'] = data['0102'].apply(logic_0102_pizang)

    # 胆囊息肉
    def logic_0102_dannangxirou(x):
        if x == x:
            if re.search(r"胆囊息肉", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['DanNangXiRou'] = data['0102'].apply(logic_0102_dannangxirou)

    # 胆囊壁毛糙
    def logic_0102_dannangbimaocao(x):
        if x == x:
            if re.search(r"胆囊壁毛糙", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['DanNangBiMaoCao'] = data['0102'].apply(logic_0102_dannangbimaocao)

    # 胆囊壁增厚
    def logic_0102_dannangbizenghou(x):
        if x == x:
            if re.search(r"胆囊壁增厚", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['DanNangBiZengHou'] = data['0102'].apply(logic_0102_dannangbizenghou)

    # 胆囊壁胆固醇结晶
    def logic_0102_dannangbidanguchun(x):
        if x == x:
            if re.search(r"胆囊壁胆固醇结晶", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['DanNangBiDanGuChun'] = data['0102'].apply(
        logic_0102_dannangbidanguchun)

    # 胆囊结石
    def logic_0102_dannangjieshi(x):
        if x == x:
            if re.search(r"胆囊结石", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['DanNangJieShi'] = data['0102'].apply(logic_0102_dannangjieshi)

    # 胆囊未探及（自述已切除）
    def logic_0102_dannangweitanji(x):
        if x == x:
            if re.search(r"胆囊未探及", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['DanNangWeiTanJi'] = data['0102'].apply(logic_0102_dannangweitanji)

    # 肾结晶肾结石
    def logic_0102_shenjieshi(x):
        if x == x:
            if re.search(r"肾结晶", str(x)):
                return 1
            elif re.search(r"肾结石", str(x)):
                return 2
            else:
                return 0
        else:
            return -1

    data['ShenJieShi'] = data['0102'].apply(logic_0102_shenjieshi)

    # 肾囊肿
    def logic_0102_shennangzhong(x):
        if x == x:
            if re.search(r"肾囊肿", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['ShenNangZhong'] = data['0102'].apply(logic_0102_shennangzhong)

    # 肾内钙化灶
    def logic_0102_shengaihuazao(x):
        if x == x:
            if re.search(r"肾内钙化灶", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['ShenNeiGaiHuaZao'] = data['0102'].apply(logic_0102_shengaihuazao)

    # 前列腺钙化灶
    def logic_0102_qianliexiangaihuazao(x):
        if x == x:
            if re.search(r"前列腺钙化灶|前列腺稍大并钙化|前列腺增生并钙化", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['QianLieXianGaiHuaZao'] = data['0102'].apply(
        logic_0102_qianliexiangaihuazao)

    # 前列腺增生
    def logic_0102_qianliexianzengsheng(x):
        if x == x:
            if re.search(r"前列腺增生|前列腺肥大|前列腺增大", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['QianLieXianZengSheng'] = data['0102'].apply(
        logic_0102_qianliexianzengsheng)

    # 前列腺囊肿
    def logic_0102_qianliexiannangzhong(x):
        if x == x:
            if re.search(r"前列腺囊肿", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['QianLieXianNangZhong'] = data['0102'].apply(
        logic_0102_qianliexiannangzhong)

    # 宫颈囊肿
    def logic_0102_gongjingnangzhong(x):
        if x == x:
            if re.search(r"宫颈囊肿", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['GongJingNangZhong'] = data['0102'].apply(
        logic_0102_gongjingnangzhong)

    # 子宫肌瘤
    def logic_0102_zigongjiliu(x):
        if x == x:
            if re.search(r"子宫肌瘤", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['ZiGongJiLiu'] = data['0102'].apply(logic_0102_zigongjiliu)

    # 绝经
    def logic_0102_juejing(x):
        if x == x:
            if re.search(r"绝经", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['JueJing'] = data['0102'].apply(logic_0102_juejing)

    # 盆腔积液
    def logic_0102_penqiangjiye(x):
        if x == x:
            if re.search(r"盆腔积液", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['PengQiangJiYe'] = data['0102'].apply(logic_0102_penqiangjiye)

    # 卵巢囊肿
    def logic_0102_luanchaonangzhong(x):
        if x == x:
            if re.search(r"卵巢囊肿", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['LuanChaoNangZhong'] = data['0102'].apply(
        logic_0102_luanchaonangzhong)

    # 内中膜
    def logic_0102_neizhongmo(x):
        if x == x:
            if re.search(r"内膜粗糙内-中膜增厚", str(x)):
                return 1
            elif re.search(r"内中膜局部增厚", str(x)):
                return 2
            elif re.search(r"内-中膜毛糙", str(x)):
                return 3
            else:
                return 0
        else:
            return -1

    data['NeiZhongMo'] = data['0102'].apply(logic_0102_neizhongmo)

    # 乳腺囊肿
    def logic_0102_ruxiannangzhong(x):
        if x == x:
            if re.search(r"乳腺囊肿", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['RuXianNangZhong'] = data['0102'].apply(logic_0102_ruxiannangzhong)

    # 甲状腺囊肿
    def logic_0102_jiazhuangxiannangzhong(x):
        if x == x:
            if re.search(r"叶囊肿", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['JiaZhuangXianNangZhong'] = data['0102'].apply(
        logic_0102_jiazhuangxiannangzhong)

    # 甲状腺囊性结节
    def logic_0102_jiazhuangxiannangjiejie(x):
        if x == x:
            if re.search(r"囊性结节", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['JiaZhuangXianNangXingJieJie'] = data['0102'].apply(
        logic_0102_jiazhuangxiannangjiejie)

    # 甲状腺结节
    def logic_0102_jiazhuangxianjiejie(x):
        if x == x:
            if re.search(r"叶结节", str(x)):
                return 1
            else:
                return 0
        else:
            return -1

    data['JiaZhuangXianJieJie'] = data['0102'].apply(
        logic_0102_jiazhuangxianjiejie)

    # 性别
    def gender_0102(x):
        man = re.search(r"前列腺", str(x))
        woman = re.search(r"乳腺|附件|子宫", str(x))
        if man:
            return 1
        elif woman:
            return 0
        else:
            return -1

    data['gender'] = data['0102'].apply(gender_0102)
    data.loc[(data['0120'].notnull()), 'gender'] = 1
    data.loc[(data['0121'].notnull()) | (data['0122'].notnull())
             | (data['0123'].notnull()), 'gender'] = 0
    data.loc[(data['0501'].notnull()) | (data['0503'].notnull()) | (data['0509'].notnull()) | (data['0516'].notnull())
             | (data['0537'].notnull()) | (data['0539'].notnull()) | (data['0541'].notnull()) | (data['0544'].notnull())
             | (data['0545'].notnull()) | (data['0546'].notnull()) | (data['0547'].notnull()) | (data['0548'].notnull())
             | (data['0549'].notnull()) | (data['0550'].notnull()) | (data['0551'].notnull()), 'gender'] = 0
    data.loc[(data['0125'].notnull()), 'gender'] = 1

    # 未加入
    # data.loc[(data['0981'].notnull()),'gender']=1
    # data.loc[(data['0982'].notnull()),'gender']=1
    # data.loc[(data['0983'].notnull()),'gender']=1
    # data.loc[(data['0984'].notnull()),'gender']=1
    # data.loc[(data['2501'].notnull()),'gender']=0

    category_feature = data[
        ['vid', 'gender', 'Medical_examination_liver', 'Medical_examination_spleen', 'Medical_history_hypertension',
         'Medical_history_hyperlipidemia',
         'Medical_history_hyperglycemia', 'Medical_history_coronary', 'Medical_history_arrhythmia',
         'Medical_examination_synthesize',
         'Medical_examination_heart_sound', 'Medical_examination_arrhythmia', 'Medical_examination_nervosum',
         'Medical_examination_respiratory_sound2', 'Medical_examination_heart_rate', 'Medical_examination_heart_voice2',
         'Medical_examination_liver_gall', 'Medical_history_hypertension_1', 'Medical_history_hyperlipidemia_1',
         'Medical_history_hyperglycemia_1', 'Medical_history_coronary_1', 'Medical_history_thyroid_1',
         'Medical_history_pregnant_1',
         'doppler_fatty_liver', 'LinBaJie', 'RuxianJiejie', 'RuxianZengsheng', 'JingmaiQuZhang', 'zhi', 'ZhiFangLiu',
         'QianLieXianZengSheng', 'BanKuai', 'FanLiu', 'gaodianya', 'JiaoMOLaoNianHuan', 'ShiWangMoDongMaiYingHua',
         'NaoDongMai',
         'DongMaiYingHua', 'GanNangZhong', 'PiZangZengDa', 'DanNangXiRou', 'DanNangBiMaoCao', 'DanNangBiZengHou',
         'DanNangBiDanGuChun', 'DanNangJieShi', 'DanNangWeiTanJi', 'ShenJieShi', 'ShenNangZhong', 'ShenNeiGaiHuaZao',
         'QianLieXianGaiHuaZao', 'QianLieXianZengSheng', 'QianLieXianNangZhong', 'GongJingNangZhong', 'ZiGongJiLiu',
         'JueJing', 'PengQiangJiYe', 'LuanChaoNangZhong', 'NeiZhongMo', 'RuXianNangZhong', 'JiaZhuangXianNangZhong',
         'JiaZhuangXianNangXingJieJie', 'JiaZhuangXianJieJie']]

    return category_feature


# 清洗训练集中的五个指标
def clean_label(x):
    x = str(x)
    if '+' in x:  # 16.04++
        i = x.index('+')
        x = x[0:i]
    if '>' in x:  # > 11.00
        i = x.index('>')
        x = x[i + 1:]
    if len(x.split('.')) > 2:  # 2.2.8
        i = x.rindex('.')
        x = x[0:i] + x[i + 1:]
    if '未做' in x or '未查' in x or '弃查' in x:
        x = np.nan
    if str(x).isdigit() == False and len(str(x)) > 4:
        x = x[0:4]
    return x


def data_clean(df):
    for c in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        df[c] = df[c].apply(clean_label)
        df[c] = df[c].astype('float64')
    return df


if __name__ == '__main__':
    # part_1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt', sep='$')
    # part_2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt', sep='$')
    # stitching(part_1,part_2)
    data = pd.read_csv('../data/data_all.csv')
    process_text(data).shape

from generation import LlaSMolGeneration
import pandas as pd
import re
    
def recording_question_answer_excel(file_path: str, llm: LlaSMolGeneration):
    data = pd.read_excel(file_path)
    col = data.columns
    data = data.values.tolist()
        
    for i, question in enumerate(data):
        res = llm.generate(question[1])
        data[i][2] = res[0]['output'][0]
        print('Question: ', question[1])
        print('Answer: ', res[0]['output'][0])
        print('----------------------------')
        if '<SMILES>' in res[0]['output'][0]:
            pattern = r"<SMILES>.*?</SMILES>"
            smiles = re.findall(pattern, res[0]['output'][0])
            smiles_clean = [i.replace('<SMILES>', '').replace('</SMILES>', '').replace(' ', '') for i in smiles]
            data[i][3] = smiles_clean
    res = pd.DataFrame(data, columns=col).to_excel(file_path, columns=col)


if __name__ == "__main__":
    generator = LlaSMolGeneration('osunlp/LlaSMol-Mistral-7B')
    file_path = "llasmol_data_4_5tasks.xlsx"
    recording_question_answer_excel(file_path, generator)
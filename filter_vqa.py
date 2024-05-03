import json
from collections import Counter

with open('/home/wuyin/COCO/VQA-CP/vqacp_v2_test_annotations.json') as f:
    answers = f.readlines()
    answers = json.loads(answers[0])


with open('/home/wuyin/COCO/VQA-CP/vqacp_v2_test_questions.json') as f:
    questions = f.readlines()
    questions = json.loads(questions[0])


qt = [e['question_type'] for e in answers]
qt = Counter(qt)

color_question = []
for ques, ans in zip(questions, answers):
    if 'color' in ans['question_type']:
        color_question.append({'question':ques, 'answer': ans})

with open('./vqa_color.jsonl', 'w') as ff:
    for row in color_question:
        ff.write(json.dumps(row))
        ff.write('\n')
print()
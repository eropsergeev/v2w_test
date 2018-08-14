from gensim.models import Word2Vec
from gensim.matutils import unitvec
import csv
import numpy as np

def get_word_vector(sentence_tokens):
    vectors_available = [word for word in sentence_tokens if word in model.wv.vocab]
    result = np.zeros((model.vector_size))
    if len(vectors_available) > 0:
        result = np.mean([model[word] for word in vectors_available], axis=0)
    return result

def tokenize(x):
	return x.split()

def get_sorted_answers(question, answers):
    question_vector = get_word_vector(tokenize(question))
    result = []
    for a in answers:
        answer_vector = get_word_vector(tokenize(a))
        similarity = np.dot(unitvec(answer_vector), unitvec(question_vector))
        result.append((a, similarity))
    return sorted(result, key=lambda x: -x[1])

def clear_text(x):
	return x.replace('?', ' ').replace(',', ' ').replace('.', ' ').replace('-', ' ').replace(':', ' ')

corr_ans = {}

with open("train.csv") as f:
	l = list(csv.reader(f))[1:]
	data = []
	qs = []
	for x in l:
		x[1] = clear_text(x[1])
		x[2 + int(x[-1])] = clear_text(x[2 + int(x[-1])])
		qs.append(x[1])
		corr_ans[x[1]] = x[2 + int(x[-1])]
		data += x[1].split() + x[2 + int(x[-1])].split()

model = Word2Vec([data])

# print(get_sorted_answers('В каком году появился YouTube', qs)[:10])
# exit(0)

res = 'question_id,correct_answer\n'

with open("test.csv") as f:
	l = list(csv.reader(f))[1:]
	for x in l[:10]:
		q_id = x[0]
		q = get_sorted_answers(clear_text(x[1]), qs)[0][0]
		anss = {clear_text(x[2 + i]) : i for i in range(3)}
		ans = get_sorted_answers(corr_ans[q], anss.keys())[0]
		res += q_id + ',' + str(anss[ans[0]]) + '\n'

with open("ans.csv", "w") as f:
	f.write(res)



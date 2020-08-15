import numpy as np
import re

class spam_classif():
    def __init__(self):
        self.spam = dict()
        self.notspam = dict()
        self.total = dict()
        self.pA = 1
        self.pNotA = 1
        self.spam_size = 1
        self.notspam_size = 1
        self.unique_word_len = 0
        self.pattern = re.compile('\w\w\w\w+')
    
    def dict_val_sum(self, dictionary):
        summa = 0
        for k, v in dictionary.items():
            summa += v
        return summa
    
    def calc_word_frequencies(self, body, label):
        try:
            body = body.lower()
            body = self.pattern.findall(body)
            for b in body:
                if b in self.total:
                    self.total[b] += 1
                else:
                    self.total[b] = 1
                        
            if label == 1:
                for b in body:
                    if b in self.spam:
                        self.spam[b] += 1
                    else:
                        self.spam[b] = 1
    
            if label == 0:
                for b in body:
                    if b in self.notspam:
                        self.notspam[b] += 1
                    else:
                        self.notspam[b] = 1
        except: pass

   
    def train(self, train_data):
        i_sp = 0
        for data in train_data:
            self.calc_word_frequencies(data[0], data[1])
            if data[1] == 1:
                i_sp += 1
        self.pA = i_sp / len(train_data)
        self.pNotA = 1 - self.pA
        
        self.spam_size = self.dict_val_sum(self.spam)
        self.notspam_size = self.dict_val_sum(self.notspam)
        self.unique_word_len = len(self.total)
    
    def calculate_P_Bi_A(self, word, label):
        if label == 1:
            if word in self.spam:            
                freq = (self.spam[word] + 1) 
            else:
                freq = 1
            return freq / (self.spam_size + self.unique_word_len)                                
        if label == 0:
            if word in self.notspam:            
                freq = (self.notspam[word] + 1) 
            else:
                freq = 1
            return freq / (self.notspam_size + self.unique_word_len)

    def calculate_P_B_A(self, text, label):
        possibility = 0
        text = text.lower()
        text = self.pattern.findall(text)
        for word in text:
            possibility += np.log(self.calculate_P_Bi_A(word,label))
        return possibility

    def classify(self, email):
        PosSpam = np.log(self.pA) + self.calculate_P_B_A(email, 1)
        PosNotSpam = np.log(self.pNotA) + self.calculate_P_B_A(email, 0)
        PosSpam_1 = 1 / (1 + np.exp(PosNotSpam - PosSpam))
        PosNotSpam_1 = 1 / (1 + np.exp(PosSpam - PosNotSpam))
        print(f'Вероятность, что это спам: {round(PosSpam_1 * 100)}%')
        print(f'Вероятность, что это не спам: {round(PosNotSpam_1 * 100)}%')
        return PosSpam_1 > PosNotSpam_1



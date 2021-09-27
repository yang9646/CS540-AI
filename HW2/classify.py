import os
import math

#  create and return training set (bag of words Python dictionary + label) from the files in a training directory
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

# create and return a vocabulary as a list of word types with counts >= cutoff in the training directory
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f, 'r', encoding='utf-8') as doc:
                # with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

# create and return a bag of words Python dictionary from a single document
def create_bow(vocab, filepath):

    bow = {}

    with open (filepath, encoding='utf-8') as doc:
        for word in doc:
            word = word.strip()
            if word in vocab:
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
            else:
                if None in bow:
                    bow[None] += 1
                else:
                    bow[None] = 1
    
    return bow

# given a training set, estimate and return the prior probability P(label) of each label
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
       
    # Adds each label from label_list into dictionary logprob
    for label in label_list:
        logprob[label] = 0
    
    # Count # labels in the training data
    for row in training_data:
        if row['label'] in label_list:
            logprob[row['label']] += 1
    
    # Calculation of log prior probability
    for label in logprob:
        numerator = logprob[label] + smooth * 1
        denominator = len(training_data) + 2
        logprob[label] = numerator / denominator
        logprob[label] = math.log(logprob[label])
        
    return logprob


# given a training set and a vocabulary, estimate and return the class conditional distribution P(word|label)
# over all words for the given label using smoothing
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    
    count_word = 0 # Given label, total word count including OOV
    
    for word in vocab:
        word_prob[word] = 0
    
    word_prob[None] = 0
    
    for row in training_data:
        # Given label
        if row['label'] == label:
            for word in row['bow']:
                # total word count including OOV
                count_word += row['bow'].get(word)
                
                if word in vocab:
                    # word count of w
                    word_prob[word] += row['bow'].get(word)
                else:
                    # word count of OOV
                    word_prob[None] += row['bow'].get(word)
    
    for word in word_prob:
        numerator = word_prob[word] + smooth * 1
        denominator = count_word + smooth * (len(vocab) + 1)
        word_prob[word] = numerator / denominator
        word_prob[word] = math.log(word_prob[word])
    
    return word_prob


# load the training data, estimate the prior distribution P(label) and class conditional distributions P(word|label),
# return the trained model
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = [f for f in os.listdir(training_directory) if not f.startswith('.')] # ignore hidden files
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab,training_directory)
    
    prior_prob = prior(training_data, ['2020', '2016'])
    con_prob_2016 = p_word_given_label(vocab, training_data, '2016')
    con_prob_2020 = p_word_given_label(vocab, training_data, '2020')
    
    retval['vocabulary'] = vocab
    retval['log prior'] = prior_prob
    retval['log p(w|y=2020)'] = con_prob_2020
    retval['log p(w|y=2016)'] = con_prob_2016
    return retval


# given a trained model, predict the label for the test document 
def classify(model, filepath):
    
    retval = {}
    
    vocab = model['vocabulary']
    prior_prob = model['log prior']
    prior_con_2020 = model['log p(w|y=2020)']
    prior_con_2016 = model['log p(w|y=2016)']
    total_2020 = 0
    total_2016 = 0

    test = create_bow(vocab, filepath)
    for word in test:
        if word in prior_con_2020.keys():
            for x in range(test.get(word)):
                total_2020 += prior_con_2020.get(word)
        if word in prior_con_2016.keys():
            for x in range(test.get(word)):
                total_2016 += prior_con_2016.get(word)
    total_2020 += prior_prob['2020']
    total_2016 += prior_prob['2016']

    retval['log p(y=2020|x)'] = total_2020
    retval['log p(y=2016|x)'] = total_2016

    if total_2020 > total_2016:
        retval['predicted y'] = '2020'
    else:
        retval['predicted y'] = '2016'
    

    return retval

###################################
# Example
# print('Classify example 1')
# model = train('./corpus/training/', 2)
# print(classify(model, './corpus/test/2016/0.txt'))

# print ('Classify example 2')
# model = train('./corpus/training/', 20)
# print(classify(model, './corpus/training/2020/1.txt'))
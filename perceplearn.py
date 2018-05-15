import string
import sys
from collections import OrderedDict
from decimal import *

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "so", "than", "too", "very", "can", "will", "just", "should", "now"]

vocabulary = set()


def read_input_vanilla():
    fin = open(sys.argv[1], 'r')
    fread = fin.read()
    lines = fread.splitlines()
    fin.close()
    global vocabulary
    weight_f = dict()
    weight_p = dict()
    class_f = dict()
    class_p = dict()

    #features = OrderedDict()
    features = dict()
    for line in lines:
        line = line.replace("'", '')
        line = line.replace('-', '')

        for p in string.punctuation:
            line = line.replace(p, ' ')

        line = line.strip(' ')
        word_list = line.split()

        if word_list[1] == 'True':
            class_f[word_list[0]] = 1
        else:
            class_f[word_list[0]] = -1

        if word_list[2] == 'Pos':
            class_p[word_list[0]] = 1
        else:
            class_p[word_list[0]] = -1

        word_count = dict()

        for i in range(3,len(word_list)):
            if word_list[i].lower() not in stopwords:
                wordLower = word_list[i].lower()
                vocabulary.add(wordLower)

                try:
                    word_count[wordLower] += 1
                except KeyError:
                    word_count[wordLower] = 1

        features[word_list[0]] = word_count

    for word in vocabulary:
        weight_f[word] = 0
        weight_p[word] = 0

    return weight_f, weight_p, class_f, class_p, features


def read_input_average():
    fin = open(sys.argv[1], 'r')
    fread = fin.read()
    lines = fread.splitlines()
    fin.close()
    global vocabulary
    weight_f = dict()
    weight_p = dict()
    cached_wf = dict()
    cached_wp = dict()
    class_f = dict()
    class_p = dict()

    features = OrderedDict()
    #features = dict()
    for line in lines:
        line = line.replace("'", '')
        line = line.replace('-', '')

        for p in string.punctuation:
            line = line.replace(p, ' ')

        line = line.strip('\n')

        word_list = line.split(' ')

        if word_list[1] == 'True':
            class_f[word_list[0]] = 1
        else:
            class_f[word_list[0]] = -1

        if word_list[2] == 'Pos':
            class_p[word_list[0]] = 1
        else:
            class_p[word_list[0]] = -1

        word_count = dict()

        for i in range(3,len(word_list)):
            if word_list[i] != '' and word_list[i].lower() not in stopwords:
                wordLower = word_list[i].lower()
                vocabulary.add(wordLower)

                try:
                    word_count[wordLower] += 1
                except KeyError:
                    word_count[wordLower] = 1

        features[word_list[0]] = word_count

    for word in vocabulary:
        weight_f[word] = 0
        weight_p[word] = 0
        cached_wf[word] = 0
        cached_wp[word] = 0

    return weight_f, weight_p, class_f, class_p, features, cached_wf, cached_wp


def vanilla_model():
    weight_f, weight_p, class_f, class_p, features = read_input_vanilla()

    bias_f = 0
    bias_p = 0

    for c in range(0,23):

        for id_val in features:
            feature = features[id_val]

            activation_f = 0
            activation_p = 0

            for word, count in feature.items():
                activation_f += (weight_f[word] * count)
                activation_p += (weight_p[word] * count)

            activation_f += bias_f
            activation_p += bias_p

            if (class_f[id_val] * activation_f) <= 0:
                for word, count in feature.items():
                    weight_f[word] += (count * class_f[id_val])

                bias_f += class_f[id_val]

            if (class_p[id_val] * activation_p) <= 0:
                for word, count in feature.items():
                    weight_p[word] += (count * class_p[id_val])

                bias_p += class_p[id_val]

    return bias_f, bias_p, weight_f, weight_p


def average_model():
    weight_f, weight_p, class_f, class_p, features, cached_wf, cached_wp = read_input_average()

    bias_f = 0
    bias_p = 0
    beta_f = 0
    beta_p = 0

    x = 1

    for c in range(0,23):

        for id_val in features:
            feature = features[id_val]

            activation_f = 0
            activation_p = 0

            for word, count in feature.items():
                activation_f += (weight_f[word] * count)
                activation_p+= (weight_p[word] * count)

            activation_f += bias_f
            activation_p += bias_p

            if (class_f[id_val] * activation_f) <= 0:
                for word, count in feature.items():
                    weight_f[word] += (count * class_f[id_val])
                    cached_wf[word] += (x * count * class_f[id_val])

                bias_f += class_f[id_val]
                beta_f += (class_f[id_val] * x)

            if (class_p[id_val] * activation_p) <= 0:
                for word, count in feature.items():
                    weight_p[word] += (count * class_p[id_val])
                    cached_wp[word] += (x * count * class_p[id_val])

                bias_p += class_p[id_val]
                beta_p += (class_p[id_val] * x)

            x += 1

    for val in weight_f:
        weight_f[val] -= Decimal(cached_wf[val]) / x

    for val in weight_p:
        weight_p[val] -= Decimal(cached_wp[val]) / x

    bias_f -= Decimal(beta_f) / x
    bias_p -= Decimal(beta_p) / x

    return bias_f, bias_p, weight_f, weight_p


def main():
    bias_f, bias_p, weight_f, weight_p = vanilla_model()

    fo = open('vanillamodel.txt', 'w+')
    fo.write("%s\n" % bias_f)
    fo.write("%s\n" % bias_p)

    for word in vocabulary:
        fo.write(word + " | weightf | %f \n" % (weight_f[word]))
        fo.write(word + " | weightp | %f \n" % (weight_p[word]))

    fo.close()

    bias_f1, bias_p1, weight_f1, weight_p1 = average_model()
    fa = open('averagedmodel.txt', 'w+')
    fa.write("%s\n" % bias_f1)
    fa.write("%s\n" % bias_p1)

    for word in vocabulary:
        fa.write(word + " | weightf | %f \n" % (weight_f1[word]))
        fa.write(word + " | weightp | %f \n" % (weight_p1[word]))

    fa.close()


if __name__ == '__main__':
    main()

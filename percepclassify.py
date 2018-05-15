import sys
from decimal import *
import string
import collections

stopwords_a = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "so", "than", "too", "very", "can", "will", "just", "should", "now"]

stopwords_v = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they","them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now", 'the', 'for', 'had', 'and', 'to', 'a', 'was', 'in', 'of', 'you', 'is', 'it', 'at', 'with', 'they', 'on', 'our', 'be', 'as', 'there', 'an', 'or', 'this', 'my', 'that']


def read_model():
    fmodel = open(sys.argv[1], 'r')
    fread = fmodel.read()
    lines = fread.splitlines()

    bias_f = Decimal(lines[0])
    bias_p = Decimal(lines[1])

    weight_f = dict()
    weight_p = dict()

    for line in lines[3:]:
        line = line.split(" ")
        word = line[0]
        weight_type = line[2]
        weight = line[4]
        if weight_type == "weightf":
            weight_f[word] = Decimal(weight)
        if weight_type == "weightp":
            weight_p[word] = Decimal(weight)

    return bias_f, bias_p, weight_f, weight_p


def perceptron_vanilla():
    bias_f, bias_p, weight_f, weight_p = read_model()
    fi = open(sys.argv[2],'r')
    fread = fi.read()
    lines = fread.splitlines()

    test_line = collections.OrderedDict()

    for line in lines:
        line = line.replace("'", '')
        line = line.replace('-', '')

        for p in string.punctuation:
            line = line.replace(p, ' ')

        line = line.strip("\n")
        word_list = line.split(" ")

        id_val = word_list[0]

        list_temp = list()
        for i in range(1,len(word_list)):
            if word_list[i] != '' and word_list[i].lower() not in stopwords_v:
                list_temp.append(word_list[i].lower())
                test_line[id_val] = list_temp

    fo = open("percepoutput.txt", 'w+')

    for key, value in test_line.items():  # k -> id_val , value -> sentence
        word_count = dict()
        for word in value:
            try:
                word_count[word] += 1
            except KeyError:
                word_count[word] = 1

        val_f = 0
        val_p = 0

        for word, count in word_count.items():
            if word in weight_f:
                val_f += weight_f[word] * count

            if word in weight_p:
                val_p += weight_p[word] * count

        val_f += bias_f
        val_p += bias_p

        if val_f > 0:
            first = 'True'
        else:
            first = 'Fake'

        if val_p > 0:
            second = 'Pos'
        else:
            second = 'Neg'

        fo.write(key + " " + first + " " + second + "\n")


def perceptron_average():
    bias_f, bias_p, weight_f, weight_p = read_model()
    fi = open(sys.argv[2],'r')
    fread = fi.read()
    lines = fread.splitlines()

    test_line = collections.OrderedDict()

    for line in lines:
        line = line.replace("'", '')
        line = line.replace('-', '')

        for p in string.punctuation:
            line = line.replace(p, ' ')

        line = line.strip("\n")
        word_list = line.split(" ")

        id_val = word_list[0]

        list_temp = list()
        for i in range(1,len(word_list)):
            if word_list[i] != '' and word_list[i].lower() not in stopwords_a:
                list_temp.append(word_list[i].lower())
                test_line[id_val] = list_temp

    fo = open("percepoutput.txt", 'w+')

    for key, value in test_line.items():  # k -> id_val , value -> sentence
        word_count = dict()
        for word in value:
            try:
                word_count[word] += 1
            except KeyError:
                word_count[word] = 1

        val_f = 0
        val_p = 0

        for word, count in word_count.items():
            if word in weight_f:
                val_f += weight_f[word] * count

            if word in weight_p:
                val_p += weight_p[word] * count

        val_f += bias_f
        val_p += bias_p

        if val_f > 0:
            first = 'True'
        else:
            first = 'Fake'

        if val_p > 0:
            second = 'Pos'
        else:
            second = 'Neg'

        fo.write(key + " " + first + " " + second + "\n")


def main():
    if sys.argv[1] == 'vanillamodel.txt':
        perceptron_vanilla()

    if sys.argv[1] == 'averagedmodel.txt':
        perceptron_average()


if __name__ == '__main__':
    main()

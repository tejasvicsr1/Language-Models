# Paths for the corpus.
import sys
technicalcorpuspath = "./Corpus/technical_domain_corpus.txt"
healthcorpuspath = "./Corpus/Health_English.txt"

# Read the data
with open(technicalcorpuspath) as f:
    techdata = f.readlines()

with open(healthcorpuspath) as f:
    healthdata = f.readlines()

# Tokenization
tokenizedtechdata = []
tokenizedhealthdata = []

for sentence in techdata:
    templist = sentence.split(" ")
    templistelen = len(templist)
    # print(templist)
    tempsentence = ""
    for i in range(0, (templistelen - 1)):
        templist[i] = templist[i].strip("\n")
        # print(templist[i])
        word = ""
        if (i == templistelen - 2):
            word = templist[i] + "."
            # print(word)
            tempsentence += word
            break
        if(len(templist[i - 1]) == 1 and ord(templist[i-1]) == 8217):
            continue
        else:
            if(len(templist[i]) == 1):
                if(ord(templist[i]) == 8217):
                    word = templist[i] + templist[i + 1]
                    i += 1
                else:
                    word = templist[i]
            else:
                word = templist [i]
        if i != templistelen - 2:
            tempsentence += word + " "
        else:
            tempsentence += word
    tokenizedtechdata.append(tempsentence)
for sentence in healthdata:
    templist = sentence.split(" ")
    templistelen = len(templist)
    # print(templist)
    tempsentence = ""
    for i in range(0, (templistelen)):
        templist[i] = templist[i].strip("\n")
        # print(templist[i])
        word = ""
        if(len(templist[i - 1]) == 1 and ord(templist[i-1]) == 8217):
            continue
        else:
            if(len(templist[i]) == 1):
                if(ord(templist[i]) == 8217):
                    word = templist[i] + templist[i + 1]
                    i += 1
                else:
                    word = templist[i]
            else:
                word = templist [i]
        if i != templistelen - 1:
            tempsentence += word + " "
        else:
            tempsentence += word
    tokenizedhealthdata.append(tempsentence)

# Splitting data
total_words_tech = 0
total_words_health = 0
trainingdata_health = tokenizedhealthdata[:-1000]
trainingdata_tech = tokenizedtechdata[:-1000]
testingdata_health = tokenizedhealthdata[-1000:]
testingdata_tech = tokenizedtechdata[-1000:]

# Finding the number of words
def numberofwords(text):
    numberofsentences = len(text)
    words = 0
    for i in range(0, numberofsentences):
        wordnumber = len(text[i])
        for j in range(0, wordnumber):
            words += 1
    return words

total_words_tech = numberofwords(trainingdata_tech)
total_words_health = numberofwords(trainingdata_health)

# Creating the ngrams for calculations.
def calculate_ngrams(text):

    listofngrams = [{}, {}, {}, {}, {}, {}]
    for gram in range(1, 5):
        for sentence in text:
            length = len(sentence) - gram
            for i in range(0, length):
                temporarygram = ' '.join(sentence[i:i+gram])
                if temporarygram not in listofngrams[gram]:
                    listofngrams[gram][temporarygram] = 1
                else:
                    listofngrams[gram][temporarygram] += 1
    return listofngrams

healthngrams = calculate_ngrams(trainingdata_health)
ngrams = calculate_ngrams(trainingdata_tech)

# print(ngrams)
# Counting frequency and numbers
def word_types(context, i):
    number = 0
    for words in ngrams[i].keys():
        if not words.startswith(context):
            pass
        else:
            number = number + 1
    return number

def health_word_types(context, i):
    number = 0
    for words in healthngrams[i].keys():
        if not words.startswith(context):
            pass
        else:
            number = number + 1
    return number

totalvalues = [0, 0, 0, 0, 0]
healthtotalvalues = [0, 0, 0, 0, 0]

# Sum all the values
for i in range(0, 5):
    ngramvalues = ngrams[i].values()
    totalvalues[i] = sum(ngramvalues)
    healthngramvalues = healthngrams[i].values()
    healthtotalvalues[i] = sum(healthngramvalues)

# Knesernay and prob of knesernay for tech corpus.
def probknesernay(sentence):
    p = 1
    d = 0.75
    def do_calculations(i,n):
        newgram = ' '.join(sentence[i-n+1:i+1])
        context = ' '.join(sentence[i-n+1:i])
        if n == 1:
            if newgram not in ngrams[n]:
                first_elem = 0
            else:
                first_elem = max(ngrams[n][newgram] - d, 0)/total_words_tech
            return first_elem + d/len(ngrams[n])
        if newgram not in ngrams[n]:
            first_elem = 0
        else:
            first_elem = max(ngrams[n][newgram] - d, 0)/ngrams[n-1][context]
        if context not in ngrams[n-1]:
            second_elem = ((d*len(ngrams[n-1]))/totalvalues[n-1]) * do_calculations(i,n-1)
        else:
            second_elem = (d/ngrams[n-1][context])*word_types(context, n) * do_calculations(i,n-1)
        return first_elem + second_elem
    end = len(sentence) - 1
    for i in range(3, end):
        p = p * do_calculations(i, 4)
    exponent = -1/len(sentence)
    return p

def knesernay(sentence):
    p = 1
    d = 0.75
    def do_calculations(i,n):
        newgram = ' '.join(sentence[i-n+1:i+1])
        context = ' '.join(sentence[i-n+1:i])
        if n == 1:
            if newgram not in ngrams[n]:
                first_elem = 0
            else:
                first_elem = max(ngrams[n][newgram] - d, 0)/total_words_tech
            return first_elem + d/len(ngrams[n])
        if newgram not in ngrams[n]:
            first_elem = 0
        else:
            first_elem = max(ngrams[n][newgram] - d, 0)/ngrams[n-1][context]
        if context not in ngrams[n-1]:
            second_elem = ((d*len(ngrams[n-1]))/totalvalues[n-1]) * do_calculations(i,n-1)
        else:
            second_elem = (d/ngrams[n-1][context])*word_types(context, n) * do_calculations(i,n-1)
        return first_elem + second_elem
    end = len(sentence) - 1
    for i in range(3, end):
        p = p * do_calculations(i, 4)
    exponent = -1/len(sentence)
    if p == 0:
        return 0
    return (p ** (exponent))
# Wittenbell and prob of witten bell for tech corpus.
def probwittenbell(sentence):
    p = 1
    d = 0.75
    def do_calculations(i, n):
        newgram = ' '.join(sentence[i-n+1:i+1])
        context = ' '.join(sentence[i-n+1:i])
        if n == 1:
            if newgram not in ngrams[n]:
                first_elem = 0
            else:
                first_elem = ngrams[n][newgram]/total_words_tech
            return first_elem + d/len(ngrams[n])
        if newgram not in ngrams[n]:
            first_elem = 0
        else:
            first_elem = ngrams[n][newgram]
        if context not in ngrams[n-1]:
            num = (d*len(ngrams[n-1]))
            dem = totalvalues[n-1]
            second_elem = (num/dem)           
            nplus = do_calculations(i, n-1)
        else:
            nplus = word_types(context, n)
            second_elem = ngrams[n-1][context]
        firstterm = first_elem + nplus*do_calculations(i, n-1)
        secondterm = second_elem + nplus
        return (firstterm)/(secondterm)
    end = len(sentence) - 1
    for i in range(3, len(sentence)-1):
        p = p * do_calculations(i, 4)
    exponent = -1/len(sentence)
    return p

def wittenbell(sentence):
    p = 1
    d = 0.75
    def do_calculations(i, n):
        newgram = ' '.join(sentence[i-n+1:i+1])
        context = ' '.join(sentence[i-n+1:i])
        if n == 1:
            if newgram not in ngrams[n]:
                first_elem = 0
            else:
                first_elem = ngrams[n][newgram]/total_words_tech
            return first_elem + d/len(ngrams[n])
        if newgram not in ngrams[n]:
            first_elem = 0
        else:
            first_elem = ngrams[n][newgram]
        if context not in ngrams[n-1]:
            num = (d*len(ngrams[n-1]))
            dem = totalvalues[n-1]
            second_elem = (num/dem)           
            nplus = do_calculations(i, n-1)
        else:
            nplus = word_types(context, n)
            second_elem = ngrams[n-1][context]
        firstterm = first_elem + nplus*do_calculations(i, n-1)
        secondterm = second_elem + nplus
        return (firstterm)/(secondterm)
    end = len(sentence) - 1
    for i in range(3, len(sentence)-1):
        p = p * do_calculations(i, 4)
    exponent = -1/len(sentence)
    if p == 0:
        return 0
    return (p ** (exponent))

# Knesernay and prob of knesernay for health corpus.
def probhealthknesernay(sentence):
    p = 1
    d = 0.75
    def do_calculations(i,n):
        newgram = ' '.join(sentence[i-n+1:i+1])
        context = ' '.join(sentence[i-n+1:i])
        if n == 1:
            if newgram not in healthngrams[n]:
                first_elem = 0
            else:
                first_elem = max(healthngrams[n][newgram] - d, 0)/total_words_health
            return first_elem + d/len(healthngrams[n])
        if newgram not in healthngrams[n]:
            first_elem = 0
        else:
            first_elem = max(healthngrams[n][newgram] - d, 0)/healthngrams[n-1][context]
        if context not in healthngrams[n-1]:
            second_elem = ((d*len(healthngrams[n-1]))/healthtotalvalues[n-1]) * do_calculations(i,n-1)
        else:
            second_elem = (d/healthngrams[n-1][context])*health_word_types(context, n) * do_calculations(i,n-1)
        return first_elem + second_elem
    end = len(sentence) - 1
    for i in range(3, end):
        p = p * do_calculations(i, 4)
    exponent = -1/len(sentence)
    return p

def healthknesernay(sentence):
    p = 1
    d = 0.75
    def do_calculations(i,n):
        newgram = ' '.join(sentence[i-n+1:i+1])
        context = ' '.join(sentence[i-n+1:i])
        if n == 1:
            if newgram not in healthngrams[n]:
                first_elem = 0
            else:
                first_elem = max(healthngrams[n][newgram] - d, 0)/total_words_health
            return first_elem + d/len(healthngrams[n])
        if newgram not in healthngrams[n]:
            first_elem = 0
        else:
            first_elem = max(healthngrams[n][newgram] - d, 0)/healthngrams[n-1][context]
        if context not in healthngrams[n-1]:
            second_elem = ((d*len(healthngrams[n-1]))/healthtotalvalues[n-1]) * do_calculations(i,n-1)
        else:
            second_elem = (d/healthngrams[n-1][context])*health_word_types(context, n) * do_calculations(i,n-1)
        return first_elem + second_elem
    end = len(sentence) - 1
    for i in range(3, end):
        p = p * do_calculations(i, 4)
    exponent = -1/len(sentence)
    if p == 0:
        return 0
    return (p ** (exponent))
# Wittenbell and prob of Wittenbell for health corpus.
def probhealthwittenbell(sentence):
    p = 1
    d = 0.75
    def do_calculations(i, n):
        newgram = ' '.join(sentence[i-n+1:i+1])
        context = ' '.join(sentence[i-n+1:i])
        if n == 1:
            if newgram not in healthngrams[n]:
                first_elem = 0
            else:
                first_elem = healthngrams[n][newgram]/total_words_health
            return first_elem + d/len(healthngrams[n])
        if newgram not in healthngrams[n]:
            first_elem = 0
        else:
            first_elem = healthngrams[n][newgram]
        if context not in healthngrams[n-1]:
            num = (d*len(healthngrams[n-1]))
            dem = healthtotalvalues[n-1]
            second_elem = (num/dem)           
            nplus = do_calculations(i, n-1)
        else:
            nplus = health_word_types(context, n)
            second_elem = healthngrams[n-1][context]
        firstterm = first_elem + nplus*do_calculations(i, n-1)
        secondterm = second_elem + nplus
        return (firstterm)/(secondterm)
    end = len(sentence) - 1
    for i in range(3, len(sentence)-1):
        p = p * do_calculations(i, 4)
    exponent = -1/len(sentence)
    return p

def healthwittenbell(sentence):
    p = 1
    d = 0.75
    def do_calculations(i, n):
        newgram = ' '.join(sentence[i-n+1:i+1])
        context = ' '.join(sentence[i-n+1:i])
        if n == 1:
            if newgram not in healthngrams[n]:
                first_elem = 0
            else:
                first_elem = healthngrams[n][newgram]/total_words_health
            return first_elem + d/len(healthngrams[n])
        if newgram not in healthngrams[n]:
            first_elem = 0
        else:
            first_elem = healthngrams[n][newgram]
        if context not in healthngrams[n-1]:
            num = (d*len(healthngrams[n-1]))
            dem = healthtotalvalues[n-1]
            second_elem = (num/dem)           
            nplus = do_calculations(i, n-1)
        else:
            nplus = health_word_types(context, n)
            second_elem = healthngrams[n-1][context]
        firstterm = first_elem + nplus*do_calculations(i, n-1)
        secondterm = second_elem + nplus
        return (firstterm)/(secondterm)
    end = len(sentence) - 1
    for i in range(3, len(sentence)-1):
        p = p * do_calculations(i, 4)
    exponent = -1/len(sentence)
    if p == 0:
        return 0
    return (p ** (exponent))

# Solution to Question 1
argumentList = sys.argv

if(len(argumentList) != 3):
    print("The input is not of the format python3 filename first_letter_of_model path_to_corpus")
    print("There are have to be exactly three arguments.")
else :
    if (argumentList[1] == 'k'):
        if(len(argumentList) == 3):
            if(argumentList[2] == './Corpus/technical_domain_corpus.txt'):
                inputsentence = str(input("input sentence: "))
                print(probknesernay(inputsentence))
            elif(argumentList[2] == './Corpus/Health_English.txt'):
                inputsentence = str(input("input sentence: "))
                print(probhealthknesernay(inputsentence))
            else:
                print("The input is not of the format python3 filename first_letter_of_model path_to_corpus")
    elif (argumentList[1] == 'w'):
        if(len(argumentList) == 3):
            if(argumentList[2] == './Corpus/technical_domain_corpus.txt'):
                inputsentence = str(input("input sentence: "))
                print(probwittenbell(inputsentence))
            elif(argumentList[2] == './Corpus/Health_English.txt'):
                inputsentence = str(input("input sentence: "))
                print(probhealthwittenbell(inputsentence))
            else:
                print("The input is not of the format python3 filename first_letter_of_model path_to_corpus")
    else:
        print("The input is not of the format python3 filename first_letter_of_model path_to_corpus")

# print(knesernay(str(input)))
# print(wittenbell(str(input)))
# print(healthknesernay(str(input)))
# print(healthwittenbell(str(input)))

# Solution to Question 2

# f = open("./Report/2019114005-LM1-train-perplexity.txt", "w")
# counter = 0
# sentencenumber = 0
# totalsum = 0
# for i in range(0, len(trainingdata_health)):
#     f.write(trainingdata_health[i] + "\t" + str(knesernay(trainingdata_health[i])) + "\n")
#     sentencenumber += 1
#     totalsum += knesernay(trainingdata_health[i])
#     counter += 1
#     if counter == 200:
#         break
# f.write("The average perplexity is " + str(totalsum/sentencenumber))
# f.close()

# f = open("./Report/2019114005-LM1-test-perplexity.txt", "w")
# counter = 0
# sentencenumber = 0
# totalsum = 0
# for i in range(0, len(testingdata_health)):
#     f.write(testingdata_health[i] + "\t" + str(knesernay(testingdata_health[i])) + "\n")
#     sentencenumber += 1
#     totalsum += knesernay(testingdata_health[i])
#     counter += 1
#     if counter == 200:
#         break
# f.write("The average perplexity is " + str(totalsum/sentencenumber))
# f.close()

# f = open("./Report/2019114005-LM2-train-perplexity.txt", "w")
# counter = 0
# sentencenumber = 0
# totalsum = 0
# for i in range(0, len(trainingdata_health)):
#     f.write(trainingdata_health[i] + "\t" + str(wittenbell(trainingdata_health[i])) + "\n")
#     sentencenumber += 1
#     totalsum += wittenbell(trainingdata_health[i])
#     counter += 1
#     if counter == 200:
#         break
# f.write("The average perplexity is " + str(totalsum/sentencenumber))
# f.close()

# f = open("./Report/2019114005-LM2-test-perplexity.txt", "w")
# counter = 0
# sentencenumber = 0
# totalsum = 0
# for i in range(0, len(testingdata_health)):
#     f.write(testingdata_health[i] + "\t" + str(wittenbell(testingdata_health[i])) + "\n")
#     sentencenumber += 1
#     totalsum += wittenbell(testingdata_health[i])
#     counter += 1
#     if counter == 200:
#         break
# f.write("The average perplexity is " + str(totalsum/sentencenumber))
# f.close()


# f = open("./Report/2019114005-LM3-train-perplexity.txt", "w")
# counter = 0
# sentencenumber = 0
# totalsum = 0
# for i in range(0, len(trainingdata_tech)):
#     f.write(trainingdata_tech[i] + "\t" + str(knesernay(trainingdata_tech[i])) + "\n")
#     sentencenumber += 1
#     totalsum += knesernay(trainingdata_tech[i])
#     counter += 1
#     if counter == 200:
#         break
# f.write("The average perplexity is " + str(totalsum/sentencenumber))
# f.close()

# f = open("./Report/2019114005-LM3-test-perplexity.txt", "w")
# counter = 0
# sentencenumber = 0
# totalsum = 0
# for i in range(0, len(testingdata_tech)):
#     f.write(testingdata_tech[i] + "\t" + str(knesernay(testingdata_tech[i])) + "\n")
#     sentencenumber += 1
#     totalsum += knesernay(testingdata_tech[i])
#     counter += 1
#     if counter == 200:
#         break
# f.write("The average perplexity is " + str(totalsum/sentencenumber))
# f.close()

# f = open("./Report/2019114005-LM4-train-perplexity.txt", "w")
# counter = 0
# sentencenumber = 0
# totalsum = 0
# for i in range(0, len(trainingdata_tech)):
#     f.write(trainingdata_tech[i] + "\t" + str(wittenbell(trainingdata_tech[i])) + "\n")
#     sentencenumber += 1
#     totalsum += wittenbell(trainingdata_tech[i])
#     counter += 1
#     if counter == 200:
#         break
# f.write("The average perplexity is " + str(totalsum/sentencenumber))
# f.close()

# f = open("./Report/2019114005-LM4-test-perplexity.txt", "w")
# counter = 0
# sentencenumber = 0
# totalsum = 0
# for i in range(0, len(testingdata_tech)):
#     f.write(testingdata_tech[i] + "\t" + str(wittenbell(testingdata_tech[i])) + "\n")
#     sentencenumber += 1
#     totalsum += wittenbell(testingdata_tech[i])
#     counter += 1
#     if counter == 200:
#         break
# f.write("The average perplexity is " + str(totalsum/sentencenumber))
# f.close()
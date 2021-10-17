# imports
import sys
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import math
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import f1_score
import nltk

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def draw_confusion(conf,label="linear"):
    plt.figure()
    plt.imshow(conf)
    plt.title("Confusion Matrix"+label)
    plt.colorbar()
    my_xticks = [i for i in range(len(conf))]
    plt.xticks(my_xticks, my_xticks)
    my_yticks = [i for i in range(len(conf))]
    plt.yticks(my_yticks, my_yticks)
    plt.set_cmap("Greens")
    plt.ylabel("True labels")
    plt.xlabel("Predicted label")
    # add points on the axis
    for i in range(len(conf)):
        for j in range(len(conf)):
            plt.text(j,i,str(conf[i,j]),ha="center",va="center",color="black")
    plt.savefig("q1_confusion_matrix"+label+".png")
    plt.show()
# Naive bayes from scratch
def naive_bayes(X,Y,Y_train_raw,vocabs,super_vocab):
    psi_i=[{},{},{},{},{}]
    psi_y=[]
    for class_label in range(total_classes):
        denom=math.log(float(len(vocabs[class_label])+len(super_vocab)))
        for word in super_vocab:
            try:
                psi_i[class_label][word]=math.log(vocabs[class_label][word]+1)-denom
            except:
                psi_i[class_label][word]=math.log(1)-denom
        psi_y.append(math.log(np.count_nonzero(np.where(Y_train_raw==class_label+1)))-math.log(len(Y_train_raw)))
    return psi_i,psi_y

def predict(x,psi_i,psi_y,super_dict_len):
    class_prob=[]
    denom=0
    for class_label in range(5):
        numer=0
        for word in x:
            try:
                numer+=psi_i[class_label][word]
            except:
                numer+=0
        p_y_x=numer+psi_y[class_label]
        class_prob.append(p_y_x)
    # class_prob.append(denom)
    class_prob=np.asarray(class_prob)
    return class_prob      
def get_predictions(X_c,psi_i,psi_y,super_dict_len):
    Y=[]
    for x in X_c:
        class_prob=predict(x,psi_i,psi_y,super_dict_len)
        Y.append(np.argmax(class_prob)+1)
    return np.array(Y)
def result_calculation(Y_train_raw,Pred_train,Y_test_raw,Pred_test,label):
    print("Training accuracy:",np.count_nonzero(np.where(Pred_train==Y_train_raw))/float(len(Pred_train)))
    print("Testing accuracy:",np.count_nonzero(np.where(Pred_test==Y_test_raw))/float(len(Pred_test)))
    result_assist(Y_train_raw,Pred_train,label+"-train")
    result_assist(Y_test_raw,Pred_test,label+"-test")
def result_assist(Y_test_raw,Pred_test,label):
    # confusion matrix
    print(label)
    conf_mat=confusion_matrix(Y_test_raw,Pred_test)
    print("Confusion Matrix")
    print(conf_mat)
    draw_confusion(conf_mat,label)
    # f1 score
    f1_matrix = f1_score(Y_test_raw,Pred_test,average=None)
    print("F1 Score")
    print(f1_matrix)
    # macro f1 score
    macro_f1 = f1_score(Y_test_raw,Pred_test,average='macro')
    print("Macro F1 Score")
    print(macro_f1)

# main function
if __name__ == '__main__':
    # get command line arguments
    train_path=sys.argv[1]
    test_path=sys.argv[2]
    part_num=sys.argv[3]
    total_classes=5
    # parameters to save computation time
    useprev=1
    useprev_tr=1
    useprev_tr_b=1
    useprev_tr_t=1
    useprev_L=1
    train_prev_q1=1
    test_prev_q1=1
    train_prev_tr_q1=1
    test_prev_tr_q1=1
    train_prev_tr_b_q1=1
    test_prev_tr_b_q1=1
    train_prev_L_q1=1
    test_prev_L_q1=1
    use_sum=1
    train_sum=1
    test_sum=1
    train_prev_tr_t_q1=1
    test_prev_tr_t_q1=1
    ngram=3
    lemma=0
    if part_num=='a':
        # read training data
        raw_data = pd.read_json(train_path,lines=True)
        Y_train_raw=np.array(raw_data['overall'])
        X_train_clean=[]
        if train_prev_q1==1:
            X_train_clean=np.load('X_train_clean_q1.npy',allow_pickle=True)
        else:
            X_train_raw=raw_data['reviewText']
            for sentence in X_train_raw:
                tokens=word_tokenize(sentence)
                tokens=[w.lower() for w in tokens]
                table=str.maketrans('','',string.punctuation)
                stripped=[w.translate(table) for w in tokens]
                words=[word for word in stripped if word.isalpha()]
                X_train_clean.append(words)
            X_train_clean=np.asarray(X_train_clean)
            np.save('X_train_clean_q1.npy',X_train_clean)
        
        X_train=[[] for i in range(total_classes)]
        for i in range(total_classes):
            X_train[i]=np.concatenate(X_train_clean[np.where(Y_train_raw==i+1)])
        Y_train=[i+1 for i in range(total_classes)]
        vocab=[{},{},{},{},{}]
        print("creating vocab")
        superDict={}
        if useprev!=1:
            for i in range(total_classes):
                unique, counts = np.unique(X_train[i], return_counts=True)
                vocab[i]=dict(zip(unique, counts))
                # save vocab as json file
                with open('vocab_'+str(i+1)+'.json', 'w') as fp:
                    json.dump(vocab[i], fp,cls=NpEncoder)
            
            print("vocab created")
            complete_words=np.concatenate(X_train)
            unique, counts = np.unique(complete_words, return_counts=True)
            super_dict=dict(zip(unique, counts))
            # save super_dict as json file
            with open('vocab_super.json', 'w') as fp:
                json.dump(super_dict, fp,cls=NpEncoder)
        else:
            for i in range(total_classes):
                with open('vocab_'+str(i+1)+'.json', 'r') as fp:
                    vocab[i]=json.load(fp)
            print("vocab loaded")
            with open('vocab_super.json', 'r') as fp:
                super_dict=json.load(fp)
            print("super dict loaded")
        # get psi_i and psi_y
        psi_i,psi_y=naive_bayes(X_train,Y_train,Y_train_raw,vocab,super_dict)
        
        # get predictions on test data
        X_test_raw=pd.read_json(test_path,lines=True)
        Y_test_raw=np.array(X_test_raw['overall'])
        if test_prev_q1==1:
            X_test_clean=np.load('X_test_clean_q1.npy',allow_pickle=True)
        else:
            X_test_clean=[]
            for sentence in X_test_raw['reviewText']:
                # tokenize sentence
                tokens=word_tokenize(sentence)
                # normalize
                tokens=[w.lower() for w in tokens]
                # remove punctuation
                table=str.maketrans('','',string.punctuation)
                stripped=[w.translate(table) for w in tokens]
                # remove remaining tokens that are not alphabetic
                words=[word for word in stripped if word.isalpha()]
                X_test_clean.append(words)
            X_test_clean=np.asarray(X_test_clean)
            np.save('X_test_clean_q1.npy',X_test_clean)
        print("Starting predictions")
        # get predictions on training data
        Pred_train=get_predictions(X_train_clean,psi_i,psi_y,len(super_dict))
        print("training accuracy:",np.count_nonzero(np.where(Pred_train==Y_train_raw))/len(Pred_train))
        Pred_test=get_predictions(X_test_clean,psi_i,psi_y,len(super_dict))
        print("testing accuracy:",np.count_nonzero(np.where(Pred_test==Y_test_raw))/len(Pred_test))


    elif part_num=='b':
        raw_data = pd.read_json(train_path,lines=True)
        Y_train_raw=np.array(raw_data['overall'])
        X_test_raw=pd.read_json(test_path,lines=True)
        Y_test_raw=np.array(X_test_raw['overall'])
        print("Starting predictions")
        # random predictions
        Pred_train=np.random.randint(1,6,size=len(Y_train_raw))
        print(np.count_nonzero(np.where(Pred_train==Y_train_raw))/len(Pred_train))
        # get predictions on test data
        Pred_test=np.random.randint(1,6,size=len(Y_test_raw))
        print(np.count_nonzero(np.where(Pred_test==Y_test_raw))/len(Pred_test))
        
        # majority prediction
        Pred_train=np.ones(len(Y_train_raw))*5
        print(np.count_nonzero(np.where(Pred_train==Y_train_raw))/len(Pred_train))
        # get predictions on test data
        Pred_test=np.ones(len(Y_test_raw))*5
        print(np.count_nonzero(np.where(Pred_test==Y_test_raw))/len(Pred_test))
        # gaussian_cvxopt(train_path,test_path)
        # tasks:
        # 1. obtain support vectors and save to file
        # 2. print validation accuracy
        # 3. print test accuracy
        # 4. print time required
    elif part_num=='c':
        raw_data = pd.read_json(train_path,lines=True)
        Y_train_raw=np.array(raw_data['overall'])
        X_train_clean=[]
        if train_prev_q1==1:
            X_train_clean=np.load('X_train_clean_q1.npy',allow_pickle=True)
        else:
            X_train_raw=raw_data['reviewText']
            for sentence in X_train_raw:
                # tokenize sentence
                tokens=word_tokenize(sentence)
                # normalize
                tokens=[w.lower() for w in tokens]
                # remove punctuation
                table=str.maketrans('','',string.punctuation)
                stripped=[w.translate(table) for w in tokens]
                # remove remaining tokens that are not alphabetic
                words=[word for word in stripped if word.isalpha()]
                X_train_clean.append(words)
            X_train_clean=np.asarray(X_train_clean)
            np.save('X_train_clean_q1.npy',X_train_clean)
        X_train=[[] for i in range(total_classes)]
        for i in range(total_classes):
            X_train[i]=np.concatenate(X_train_clean[np.where(Y_train_raw==i+1)])
        Y_train=[i+1 for i in range(total_classes)]
        vocab=[{},{},{},{},{}]
        print("creating vocab")
        superDict={}
        if useprev!=1:
            for i in range(total_classes):
                unique, counts = np.unique(X_train[i], return_counts=True)
                vocab[i]=dict(zip(unique, counts))
                # save vocab as json file
                with open('vocab_'+str(i+1)+'.json', 'w') as fp:
                    json.dump(vocab[i], fp,cls=NpEncoder)
            
            print("vocab created")
            complete_words=np.concatenate(X_train)
            unique, counts = np.unique(complete_words, return_counts=True)
            super_dict=dict(zip(unique, counts))
            # save super_dict as json file
            with open('vocab_super.json', 'w') as fp:
                json.dump(super_dict, fp,cls=NpEncoder)
        else:
            for i in range(total_classes):
                with open('vocab_'+str(i+1)+'.json', 'r') as fp:
                    vocab[i]=json.load(fp)
            print("vocab loaded")
            with open('vocab_super.json', 'r') as fp:
                super_dict=json.load(fp)
            print("super dict loaded")
        # get psi_i and psi_y
        psi_i,psi_y=naive_bayes(X_train,Y_train,Y_train_raw,vocab,super_dict)
        
        # get predictions on test data
        X_test_raw=pd.read_json(test_path,lines=True)
        Y_test_raw=np.array(X_test_raw['overall'])
        if test_prev_q1==1:
            X_test_clean=np.load('X_test_clean_q1.npy',allow_pickle=True)
        else:
            X_test_clean=[]
            for sentence in X_test_raw['reviewText']:
                # tokenize sentence
                tokens=word_tokenize(sentence)
                # normalize
                tokens=[w.lower() for w in tokens]
                # remove punctuation
                table=str.maketrans('','',string.punctuation)
                stripped=[w.translate(table) for w in tokens]
                # remove remaining tokens that are not alphabetic
                words=[word for word in stripped if word.isalpha()]
                X_test_clean.append(words)
            X_test_clean=np.asarray(X_test_clean)
            np.save('X_test_clean_q1.npy',X_test_clean)
        print("Starting predictions")
        Pred_test=get_predictions(X_test_clean,psi_i,psi_y,len(super_dict))
        print(np.count_nonzero(np.where(Pred_test==Y_test_raw))/len(Pred_test))


        # confusion matrix
        conf_mat=confusion_matrix(Y_test_raw,Pred_test)
        f1_matrix = f1_score(Y_test_raw,Pred_test,average=None)
        print("F1 Score")
        print(f1_matrix)
        print("Confusion Matrix")
        print(conf_mat)
        macro_f1 = f1_score(Y_test_raw,Pred_test,average='macro')
        print("Macro F1 Score")
        print(macro_f1)
        draw_confusion(conf_mat,"nb-basic")

        print(0)
        # libsvm_linear_gaussian(train_path,test_path)
        # tasks:
        # 1. obtain support vectors and save to file
        # 2. print bias
        # 3. print weights
        # 4. print validation accuracy
        # 5. print test accuracy
        # 6. print time required
    elif part_num=='d':
        # read training data
        raw_data = pd.read_json(train_path,lines=True)
        Y_train_raw=np.array(raw_data['overall'])
        X_train_clean=[]
        if train_prev_tr_q1==1:
            X_train_clean_tr=np.load('X_train_clean_tr_q1.npy',allow_pickle=True)
        else:
            X_train_clean=np.load('X_train_clean_q1.npy',allow_pickle=True)
            X_train_clean_tr=[]
            sw_nltk=set(stopwords.words('english'))
            for sentence in X_train_clean:
                words=[w for w in sentence if w not in sw_nltk]
                ps=PorterStemmer()
                words=[ps.stem(w) for w in words]
                X_train_clean_tr.append(words)
            X_train_clean_tr=np.asarray(X_train_clean_tr)
            np.save('X_train_clean_tr_q1.npy',X_train_clean_tr)
        X_train=[[] for i in range(total_classes)]
        for i in range(total_classes):
            X_train[i]=np.concatenate(X_train_clean_tr[np.where(Y_train_raw==i+1)])
        Y_train=[i+1 for i in range(total_classes)]
        vocab=[{},{},{},{},{}]
        print("creating vocab")
        superDict={}
        if useprev_tr!=1:
            for i in range(total_classes):
                unique, counts = np.unique(X_train[i], return_counts=True)
                vocab[i]=dict(zip(unique, counts))
                # save vocab as json file
                with open('vocab_'+str(i+1)+'_tr.json', 'w') as fp:
                    json.dump(vocab[i], fp,cls=NpEncoder)
            
            print("vocab created")
            complete_words=np.concatenate(X_train)
            unique, counts = np.unique(complete_words, return_counts=True)
            super_dict=dict(zip(unique, counts))
            # save super_dict as json file
            with open('vocab_super_tr.json', 'w') as fp:
                json.dump(super_dict, fp,cls=NpEncoder)
        else:
            for i in range(total_classes):
                with open('vocab_'+str(i+1)+'_tr.json', 'r') as fp:
                    vocab[i]=json.load(fp)
            print("vocab loaded")
            with open('vocab_super_tr.json', 'r') as fp:
                super_dict=json.load(fp)
            print("super dict loaded")
        # get psi_i and psi_y
        psi_i,psi_y=naive_bayes(X_train,Y_train,Y_train_raw,vocab,super_dict)
        
        # get predictions on test data
        X_test_raw=pd.read_json(test_path,lines=True)
        Y_test_raw=np.array(X_test_raw['overall'])
        if test_prev_tr_q1==1:
            X_test_clean_tr=np.load('X_test_clean_tr_q1.npy',allow_pickle=True)
        else:
            X_test_clean=np.load('X_test_clean_q1.npy',allow_pickle=True)
            X_test_clean_tr=[]
            sw_nltk=set(stopwords.words('english'))
            for sentence in X_test_clean:
                words=[w for w in sentence if w not in sw_nltk]
                ps=PorterStemmer()
                words=[ps.stem(w) for w in words]
                X_test_clean_tr.append(words)
            X_test_clean_tr=np.asarray(X_test_clean_tr)
            np.save('X_test_clean_tr_q1.npy',X_test_clean_tr)
        print("Starting predictions")
        Pred_test=get_predictions(X_test_clean_tr,psi_i,psi_y,len(super_dict))
        Pred_train=get_predictions(X_train_clean_tr,psi_i,psi_y,len(super_dict))
        result_calculation(Y_train_raw,Pred_train,Y_test_raw,Pred_test,"stem")
    elif part_num=='e':
        # read training data
        if ngram==2:
            raw_data = pd.read_json(train_path,lines=True)
            Y_train_raw=np.array(raw_data['overall'])
            X_train_clean=[]
            if train_prev_tr_b_q1==1:
                X_train_clean_tr=np.load('X_train_clean_tr_bi_q1.npy',allow_pickle=True)
            else:
                X_train_clean=np.load('X_train_clean_tr_q1.npy',allow_pickle=True)
                X_train_clean_tr=[]
                sw_nltk=set(stopwords.words('english'))
                for sentence in X_train_clean:
                    bigram=list(nltk.ngrams(sentence,2))
                    bigrams=[str(i[0])+' '+str(i[1]) for i in bigram]
                    X_train_clean_tr.append(bigrams)
                X_train_clean_tr=np.asarray(X_train_clean_tr)
                np.save('X_train_clean_tr_bi_q1.npy',X_train_clean_tr)
            X_train=[[] for i in range(total_classes)]
            for i in range(total_classes):
                X_train[i]=np.concatenate(X_train_clean_tr[np.where(Y_train_raw==i+1)])
            Y_train=[i+1 for i in range(total_classes)]
            vocab=[{},{},{},{},{}]
            print("creating vocab")
            superDict={}
            if useprev_tr_b!=1:
                for i in range(total_classes):
                    unique, counts = np.unique(X_train[i], return_counts=True)
                    vocab[i]=dict(zip(unique, counts))
                    # save vocab as json file
                    with open('vocab_'+str(i+1)+'_tr_b.json', 'w') as fp:
                        json.dump(vocab[i], fp,cls=NpEncoder)
                
                print("vocab created")
                complete_words=np.concatenate(X_train)
                unique, counts = np.unique(complete_words, return_counts=True)
                super_dict=dict(zip(unique, counts))
                # save super_dict as json file
                with open('vocab_super_tr_b.json', 'w') as fp:
                    json.dump(super_dict, fp,cls=NpEncoder)
            else:
                for i in range(total_classes):
                    with open('vocab_'+str(i+1)+'_tr_b.json', 'r') as fp:
                        vocab[i]=json.load(fp)
                print("vocab loaded")
                with open('vocab_super_tr_b.json', 'r') as fp:
                    super_dict=json.load(fp)
                print("super dict loaded")
            # get psi_i and psi_y
            psi_i,psi_y=naive_bayes(X_train,Y_train,Y_train_raw,vocab,super_dict)
            
            # get predictions on test data
            X_test_raw=pd.read_json(test_path,lines=True)
            Y_test_raw=np.array(X_test_raw['overall'])
            if test_prev_tr_b_q1==1:
                X_test_clean_tr=np.load('X_test_clean_tr_bi_q1.npy',allow_pickle=True)
            else:
                X_test_clean=np.load('X_test_clean_tr_q1.npy',allow_pickle=True)
                X_test_clean_tr=[]
                sw_nltk=set(stopwords.words('english'))
                for sentence in X_test_clean:
                    bigram=list(nltk.bigrams(sentence))
                    bigrams=[str(i[0])+' '+str(i[1]) for i in bigram]
                    X_test_clean_tr.append(bigrams)
                X_test_clean_tr=np.asarray(X_test_clean_tr)
                np.save('X_test_clean_tr_bi_q1.npy',X_test_clean_tr)
        else:
            raw_data = pd.read_json(train_path,lines=True)
            Y_train_raw=np.array(raw_data['overall'])
            X_train_clean=[]
            if train_prev_tr_t_q1==1:
                X_train_clean_tr=np.load('X_train_clean_tr_tri_q1.npy',allow_pickle=True)
            else:
                X_train_clean=np.load('X_train_clean_tr_q1.npy',allow_pickle=True)
                X_train_clean_tr=[]
                sw_nltk=set(stopwords.words('english'))
                for sentence in X_train_clean:
                    bigram=list(nltk.ngrams(sentence,3))
                    bigrams=[str(i[0])+' '+str(i[1]) for i in bigram]
                    X_train_clean_tr.append(bigrams)
                X_train_clean_tr=np.asarray(X_train_clean_tr)
                np.save('X_train_clean_tr_tri_q1.npy',X_train_clean_tr)
            X_train=[[] for i in range(total_classes)]
            for i in range(total_classes):
                X_train[i]=np.concatenate(X_train_clean_tr[np.where(Y_train_raw==i+1)])
            Y_train=[i+1 for i in range(total_classes)]
            vocab=[{},{},{},{},{}]
            print("creating vocab")
            superDict={}
            if useprev_tr_t!=1:
                for i in range(total_classes):
                    unique, counts = np.unique(X_train[i], return_counts=True)
                    vocab[i]=dict(zip(unique, counts))
                    # save vocab as json file
                    with open('vocab_'+str(i+1)+'_tr_tri.json', 'w') as fp:
                        json.dump(vocab[i], fp,cls=NpEncoder)
                
                print("vocab created")
                complete_words=np.concatenate(X_train)
                unique, counts = np.unique(complete_words, return_counts=True)
                super_dict=dict(zip(unique, counts))
                # save super_dict as json file
                with open('vocab_super_tr_tri.json', 'w') as fp:
                    json.dump(super_dict, fp,cls=NpEncoder)
            else:
                for i in range(total_classes):
                    with open('vocab_'+str(i+1)+'_tr_tri.json', 'r') as fp:
                        vocab[i]=json.load(fp)
                print("vocab loaded")
                with open('vocab_super_tr_tri.json', 'r') as fp:
                    super_dict=json.load(fp)
                print("super dict loaded")
            # get psi_i and psi_y
            psi_i,psi_y=naive_bayes(X_train,Y_train,Y_train_raw,vocab,super_dict)
            
            # get predictions on test data
            X_test_raw=pd.read_json(test_path,lines=True)
            Y_test_raw=np.array(X_test_raw['overall'])
            if test_prev_tr_t_q1==1:
                X_test_clean_tr=np.load('X_test_clean_tr_tri_q1.npy',allow_pickle=True)
            else:
                X_test_clean=np.load('X_test_clean_tr_q1.npy',allow_pickle=True)
                X_test_clean_tr=[]
                sw_nltk=set(stopwords.words('english'))
                for sentence in X_test_clean:
                    bigram=list(nltk.bigrams(sentence))
                    bigrams=[str(i[0])+' '+str(i[1]) for i in bigram]
                    X_test_clean_tr.append(bigrams)
                X_test_clean_tr=np.asarray(X_test_clean_tr)
                np.save('X_test_clean_tr_tri_q1.npy',X_test_clean_tr)
        print("Starting predictions")
        Pred_test=get_predictions(X_test_clean_tr,psi_i,psi_y,len(super_dict))
        Pred_train=get_predictions(X_train_clean_tr,psi_i,psi_y,len(super_dict))
        result_calculation(Y_train_raw,Pred_train,Y_test_raw,Pred_test,str(ngram)+"gram")

    elif part_num=='f':
        if lemma==1:
            from nltk.stem import WordNetLemmatizer
            raw_data = pd.read_json(train_path,lines=True)
            Y_train_raw=np.array(raw_data['overall'])
            X_train_clean=[]
            if train_prev_L_q1==1:
                X_train_clean_tr=np.load('X_train_clean_L_q1.npy',allow_pickle=True)
            else:
                X_train_clean=np.load('X_train_clean_q1.npy',allow_pickle=True)
                X_train_clean_tr=[]
                lemmatizer=WordNetLemmatizer()
                sw_nltk=set(stopwords.words('english'))
                for sentence in X_train_clean:
                    words=[w for w in sentence if w not in sw_nltk]
                    words=[lemmatizer.lemmatize(word) for word in words]
                    X_train_clean_tr.append(words)
                X_train_clean_tr=np.asarray(X_train_clean_tr)
                np.save('X_train_clean_L_q1.npy',X_train_clean_tr)
            X_train=[[] for i in range(total_classes)]
            for i in range(total_classes):
                X_train[i]=np.concatenate(X_train_clean_tr[np.where(Y_train_raw==i+1)])
            Y_train=[i+1 for i in range(total_classes)]
            vocab=[{},{},{},{},{}]
            print("creating vocab")
            superDict={}
            if useprev_L!=1:
                for i in range(total_classes):
                    unique, counts = np.unique(X_train[i], return_counts=True)
                    vocab[i]=dict(zip(unique, counts))
                    # save vocab as json file
                    with open('vocab_'+str(i+1)+'_L.json', 'w') as fp:
                        json.dump(vocab[i], fp,cls=NpEncoder)
                
                print("vocab created")
                complete_words=np.concatenate(X_train)
                unique, counts = np.unique(complete_words, return_counts=True)
                super_dict=dict(zip(unique, counts))
                # save super_dict as json file
                with open('vocab_super_L.json', 'w') as fp:
                    json.dump(super_dict, fp,cls=NpEncoder)
            else:
                for i in range(total_classes):
                    with open('vocab_'+str(i+1)+'_L.json', 'r') as fp:
                        vocab[i]=json.load(fp)
                print("vocab loaded")
                with open('vocab_super_L.json', 'r') as fp:
                    super_dict=json.load(fp)
                print("super dict loaded")
            # get psi_i and psi_y
            psi_i,psi_y=naive_bayes(X_train,Y_train,Y_train_raw,vocab,super_dict)
            
            # get predictions on test data
            X_test_raw=pd.read_json(test_path,lines=True)
            Y_test_raw=np.array(X_test_raw['overall'])
            if test_prev_L_q1==1:
                X_test_clean_tr=np.load('X_test_clean_L_q1.npy',allow_pickle=True)
            else:
                X_test_clean=np.load('X_test_clean_q1.npy',allow_pickle=True)
                X_test_clean_tr=[]
                lemmatizer=WordNetLemmatizer()
                sw_nltk=set(stopwords.words('english'))
                for sentence in X_test_clean:
                    words=[w for w in sentence if w not in sw_nltk]
                    words=[lemmatizer.lemmatize(word) for word in words]
                    X_test_clean_tr.append(words)
                X_test_clean_tr=np.asarray(X_test_clean_tr)
                np.save('X_test_clean_L_q1.npy',X_test_clean_tr)
            print("Starting predictions")
            Pred_test=get_predictions(X_test_clean_tr,psi_i,psi_y,len(super_dict))
            Pred_train=get_predictions(X_train_clean_tr,psi_i,psi_y,len(super_dict))
            result_calculation(Y_train_raw,Pred_train,Y_test_raw,Pred_test,"lemm")
        else:
            import nltk
            from nltk import word_tokenize
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            print("Loading data")
            raw_data = pd.read_json(train_path,lines=True)
            Y_train=np.array(raw_data['overall'])
            X_train=np.array(raw_data['reviewText'])
            raw_data=pd.read_json(test_path,lines=True)
            Y_test=np.array(raw_data['overall'])
            X_test=np.array(raw_data['reviewText'])
            vectorizer = TfidfVectorizer(preprocessor=None,tokenizer=word_tokenize, analyzer='word', stop_words=None, strip_accents=None, lowercase=True,ngram_range=(1,3), min_df=0.0001, max_df=0.9,binary=False,norm='l2',use_idf=1,smooth_idf=1, sublinear_tf=1)
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)
            print("Starting training")
            mnb = MultinomialNB()
            mnb.fit(X_train,Y_train)
            print("Starting predictions")
            predmnb = mnb.predict(X_test)
            print("Feature Engineering Score:",round(accuracy_score(Y_test,predmnb),5));
            print("Macro F1 score =", f1_score(predmnb, Y_test, average='macro')) 
    elif part_num=='g':
        # read training data
        raw_data = pd.read_json(train_path,lines=True)
        Y_train_raw=np.array(raw_data['overall'])
        X_train_clean=[]
        if train_sum==1:
            X_train_clean=np.load('X_train_sum_clean_q1.npy',allow_pickle=True)
        else:
            X_train_raw=raw_data['summary']
            sw_nltk=set(stopwords.words('english'))
            ps=PorterStemmer()
            for sentence in X_train_raw:
                # tokenize sentence
                tokens=word_tokenize(sentence)
                # normalize
                tokens=[w.lower() for w in tokens]
                # remove punctuation
                table=str.maketrans('','',string.punctuation)
                stripped=[w.translate(table) for w in tokens]
                # remove remaining tokens that are not alphabetic
                words=[word for word in stripped if word.isalpha()]
                # filter out stop words
                words=[w for w in words if w not in sw_nltk]
                words=[ps.stem(w) for w in words]
                X_train_clean.append(words)
            X_train_clean=np.asarray(X_train_clean)
            np.save('X_train_sum_clean_q1.npy',X_train_clean)
        X_train=[[] for i in range(total_classes)]
        for i in range(total_classes):
            X_train[i]=np.concatenate(X_train_clean[np.where(Y_train_raw==i+1)])
        Y_train=[i+1 for i in range(total_classes)]
        vocab=[{},{},{},{},{}]
        print("creating vocab")
        superDict={}
        if use_sum!=1:
            for i in range(total_classes):
                unique, counts = np.unique(X_train[i], return_counts=True)
                vocab[i]=dict(zip(unique, counts))
                # save vocab as json file
                with open('vocab_'+str(i+1)+'_sum.json', 'w') as fp:
                    json.dump(vocab[i], fp,cls=NpEncoder)
            
            print("vocab created")
            complete_words=np.concatenate(X_train)
            unique, counts = np.unique(complete_words, return_counts=True)
            super_dict=dict(zip(unique, counts))
            # save super_dict as json file
            with open('vocab_super_sum.json', 'w') as fp:
                json.dump(super_dict, fp,cls=NpEncoder)
        else:
            for i in range(total_classes):
                with open('vocab_'+str(i+1)+'_sum.json', 'r') as fp:
                    vocab[i]=json.load(fp)
            print("vocab loaded")
            with open('vocab_super_sum.json', 'r') as fp:
                super_dict=json.load(fp)
            print("super dict loaded")
        # get psi_i and psi_y
        psi_i,psi_y=naive_bayes(X_train,Y_train,Y_train_raw,vocab,super_dict)
        
        # get predictions on test data
        X_test_raw=pd.read_json(test_path,lines=True)
        Y_test_raw=np.array(X_test_raw['overall'])
        if test_sum==1:
            X_test_clean=np.load('X_test_sum_clean_q1.npy',allow_pickle=True)
        else:
            X_test_clean=[]
            ps=PorterStemmer()
            
            for sentence in X_test_raw['summary']:
                # tokenize sentence
                tokens=word_tokenize(sentence)
                # normalize
                tokens=[w.lower() for w in tokens]
                # remove punctuation
                table=str.maketrans('','',string.punctuation)
                stripped=[w.translate(table) for w in tokens]
                # remove remaining tokens that are not alphabetic
                words=[word for word in stripped if word.isalpha()]
                # filter out stop words
                words=[w for w in words if w not in sw_nltk]
                words=[ps.stem(w) for w in words]
                X_test_clean.append(words)
            X_test_clean=np.asarray(X_test_clean)
            np.save('X_test_sum_clean_q1.npy',X_test_clean)
        print("Starting predictions")
        Pred_test=get_predictions(X_test_clean,psi_i,psi_y,len(super_dict))
        Pred_train=get_predictions(X_train_clean,psi_i,psi_y,len(super_dict))
        result_calculation(Y_train_raw,Pred_train,Y_test_raw,Pred_test,"summary")


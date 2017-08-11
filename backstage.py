from data_preprocessor import *
from AutoRec import AutoRec
import tensorflow as tf
import time
import argparse
from flask import Flask,render_template,request,flash,json
from models import  Movie,Movies,LoginForm
from recom import *
import pickle
app = Flask(__name__)
app.secret_key =  '123'
current_time = time.time()

@app.route('/',methods=['GET','POST'])
def hello_world():
    path1="./data/ml-1m/"
    if request.method == 'POST':
        movies = []
        err=0
        ided=[]
        #a = request.values.get("movie1")
        score1=score2=score3=score4=score5=score6=score7=score8=score9=score10=score11=score12=score13=score14=score15=score16=score17=score18=score19=score20=score21=score22=score23=score24=score25=score26=score27=score28=score29=score30=score31=score32=score33=score34=score35=score36=score37=score38=score39=score40=score41=score42=score43=score44=score45=score46=score47=score48=score49=score50=0
        score=locals()
        for i in range(1,51):
            if request.form['score%s' %i]:
                score['score%s' %i] = float(request.form['score%s' %i])
                if (score['score%s' % i] > 5) or (score['score%s' % i] < 1): flash("请输入正确范围的评分");err=1
        if err == 1:
            a = '1~5分'
            return render_template("index2.html")
        for j in range(1,51):
            if score['score%s' %j]!=0 and score['score%s' %j]<=5 and score['score%s' %j]>0:
                movie = '%d\t%d\t%d\t%d\n' % (944, j, score['score%s' %j], 0)
                movies.append(movie)
                ided.append(j)
        #print(ided)
        data = []
        for line in open(path1 + '/u.data'):
            (user, movieid, rating, ts) = line.split('\t')
            a = '%d\t%d\t%d\t%d\n'% (int(user), int(movieid), int(rating), int(ts))
            data.append(a)
        output = open(path1+'/result.txt','w')
        output.writelines(data)
        output.writelines(movies)
        output.close()
        f=open(path1+'/result.txt')
        lines = f.readlines()
        num=len(lines)
        print(num)

        parser = argparse.ArgumentParser(description='custom AutoRec ')
        parser.add_argument('--train_epoch', type=int, default=100)
        parser.add_argument('--display_step', type=int, default=1)
        parser.add_argument('--lr', type=float, default=5e-3)
        parser.add_argument('--lambda_value', type=float, default=1)
        parser.add_argument('--random_seed', type=int, default=100)
        parser.add_argument('--optimizer_method',
                            choices=['Adam', 'Adadelta', 'Adagrad', 'RMSProp', 'GradientDescent', 'Momentum'],
                            default='Adam')
        parser.add_argument('--g_act', choices=['Sigmoid', 'Relu', 'Elu', 'Tanh', "Identity"], default='Sigmoid')
        parser.add_argument('--f_act', choices=['Sigmoid', 'Relu', 'Elu', 'Tanh', "Identity"], default='Identity')

        args = parser.parse_args()

        data_name = 'ml-1m'
        path = "./data/%s" % data_name + "/"
        num_users = 944
        num_items = 1682
        num_total_ratings = num

        train_ratio = 0.9
        hidden_neuron = 500
        random_seed = args.random_seed
        batch_size = 256
        lr = args.lr
        train_epoch = args.train_epoch
        optimizer_method = args.optimizer_method
        # optimizer_method = 'RMSProp'
        display_step = args.display_step
        decay_epoch_step = 10
        lambda_value = args.lambda_value

        if args.f_act == "Sigmoid":
            f_act = tf.nn.sigmoid
        elif args.f_act == "Relu":
            f_act = tf.nn.relu
        elif args.f_act == "Tanh":
            f_act = tf.nn.tanh
        elif args.f_act == "Identity":
            f_act = tf.identity
        elif args.f_act == "Elu":
            f_act = tf.nn.elu
        else:
            raise NotImplementedError("ERROR")

        if args.g_act == "Sigmoid":
            g_act = tf.nn.sigmoid
        elif args.g_act == "Relu":
            g_act = tf.nn.relu
        elif args.g_act == "Tanh":
            g_act = tf.nn.tanh
        elif args.g_act == "Identity":
            g_act = tf.identity
        elif args.g_act == "Elu":
            g_act = tf.nn.elu
        else:
            raise NotImplementedError("ERROR")

        date = "0325"
        result_path = './results/' + data_name + '/' + date + '/' + str(random_seed) + '_' + str(
            optimizer_method) + '_' + str(lr) + "_" + str(current_time) + "/"

        movies1,prefs1, R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, \
        user_train_set, item_train_set, user_test_set, item_test_set \
            = read_rating(path, num_users, num_items, num_total_ratings, 1, 0, train_ratio, random_seed)
        #print(movies1)
        try:
            with tf.Session() as sess:
                from AutoRec import AutoRec
                AutoRec = AutoRec(sess, args,
                                  num_users, num_items, hidden_neuron, f_act, g_act,
                                  R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings,
                                  num_test_ratings,
                                  train_epoch, batch_size, lr, optimizer_method, display_step, random_seed,
                                  decay_epoch_step, lambda_value,
                                  user_train_set, item_train_set, user_test_set, item_test_set,
                                  result_path, date, data_name, movies1)
                AutoRec.run()
        except :
            print("error")

        #movie = open("./results/recom.txt")
        movie2={}
        for line in open( './results/recom.txt'):
            (rating, name,id) = line.split('\t')
            if id in ided:continue
            movie2[name] = rating
        #print(movie2)
        rankings = [(rating, item) for item, rating in movie2.items()]
        rankings.sort()
        rankings.reverse()

        #print(a)
        # prefs = loadMovieLens()
        # pkl = open('C:/Users/computer/Desktop/biyesheji/程序/datacos.pkl', 'rb')
        # data2 = pickle.load(pkl)
        # wupin = hunhe(prefs, data2, '8888')
        #ten1 = ten(a)
        new_dict = {}
        j=0
        for i in rankings:
            new_dict[i[1]] = i[0]
            if j==10:break
            j += 1
        a=[]
        for name in new_dict:
            a.append(name)
        #print(a)

        return render_template("index3.html",movie1=a[0],movie2=a[1],movie3=a[2],movie4=a[3],movie5=a[4],movie6=a[5],movie7=a[6],movie8=a[7],movie9=a[8],movie10=a[9])
    return render_template("index2.html")

if __name__ == '__main__':
    app.run()

#coding:utf-8
#MovieLens的数据集
from math import sqrt
from models import  Movie
import pickle

pkl = open('C:/Users/computer/Desktop/biyesheji/程序/datacos.pkl', 'rb')
data2 = pickle.load(pkl)

def loadMovieLens(path='C:/Users/computer/Desktop/biyesheji/data'):
    #获取电影的标题
    movies={}
    for line in open(path+'/u.item',encoding='utf-8'):
        (id,title)=line.split('|')[0:2]
        movies[id]=title


    #加载数据
    prefs={}
    for line in open(path+'/result.txt'):
        (user,movieid,rating,ts)=line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movies[movieid]]=float(rating)
    return prefs
prefs=loadMovieLens()

#欧几里得距离
def sim_distance(prefs,person1,person2):
    si={}#share_item的列表
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1

    #如果没有共同爱好，返回0
    if len(si)==0:return 0
    #计算所有差值的平方和
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2)for item in prefs[person1] if item in prefs[person2]])
    return 1/(1+sqrt(sum_of_squares))
#print(sim_distance(prefs,'1','222'))
#皮尔逊相关度评价
def sim_person(prefs,p1,p2):
    #找到双方都评价过的物品
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]:si[item]=1
    #得到列表元素的个数
    n=len(si)
    #若没有共同评价过的物品，返回1,防止除数为0错误
    if n==0:return 1
    #对所有偏好求和，平方，平方和，乘积和，计算皮尔逊相关度
    sum1=sum([prefs[p1][it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])
    sum1sq = sum([pow(prefs[p1][it],2) for it in si])
    sum2sq = sum([pow(prefs[p2][it],2) for it in si])
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si ])
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1sq-pow(sum1,2)/n)*(sum2sq-pow(sum2,2)/n))
    if den==0: return 0
    r=num/den
    return r
def sim_pearson(prefs,p1,p2,p):
    #找到双方都评价过的物品

    si={}
    for item in p:
        if item in prefs[p2]:si[item]=1
    #得到列表元素的个数
    n=len(si)
    #若没有共同评价过的物品，返回1,防止除数为0错误
    if n==0:return 1
    #对所有偏好求和，平方，平方和，乘积和，计算皮尔逊相关度
    sum1=sum([p[it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])
    sum1sq = sum([pow(p[it],2) for it in si])
    sum2sq = sum([pow(prefs[p2][it],2) for it in si])
    pSum=sum([p[it]*prefs[p2][it] for it in si ])
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1sq-pow(sum1,2)/n)*(sum2sq-pow(sum2,2)/n))
    if den==0: return 0
    r=num/den
    return r


#tonimoto相似度
def sim_tonimoto(user_data, user1, user2):
    common = {}
    # 判断有没有相同的数据, 没有相同数据则返回0
    for item in user_data[user1]:
        if item in user_data[user2]:
            common[item] = 1
    if len(common) == 0:
        return 0
    common_num = len(common)
    user1_num = len(user_data[user1])
    user2_num = len(user_data[user2])
    res = float(common_num) / (user1_num + user2_num - common_num)
    return res
#print(sim_tonimoto(prefs,'1','222'))

#物品相似度余弦算法
def CosSim(item_tags,i,j):
    ret  = 0
    for b,wib in item_tags[i].items():
        if b in item_tags[j]:
            ret += wib * item_tags[j][b]
    ni = 0
    nj = 0
    for b,w in item_tags[i].items():
        ni += w * w
    for b,w in item_tags[j].items():
        nj += w * w
    if ret == 0:
        return 0
    return ret / sqrt(ni * nj)
#print(CosSim(prefs,'1','222'))
#找相识评论者
def topMatches(prefs,person,n=5,similarity=CosSim):
    scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


#推荐物品
def getRecommendations(prefs,person,similarity=sim_person):
    totals={}
    simSums={}
    for other in prefs:
        #不和自己比较
        if other==person: continue
        simm=similarity(prefs,person,other)
        #忽略评价值为0或者小于0的情况
        #print(simm)
        if simm<=0: continue
        if simm>0:
            for item in prefs[other]:
                # 只对自己没看过的电影评价
                if item not in prefs[person] or prefs[person][item] == 0:
                    # 相识度乘以评价值
                    totals.setdefault(item, 0)
                    totals[item] += prefs[other][item] * simm
                    # 相识度之和
                    simSums.setdefault(item, 0)
                    simSums[item] += simm
    #建议一个归一化的列表
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    dir = {}
    for i, j in rankings:
        dir.setdefault(j, 0)
        dir[j] = i
    return dir
a=getRecommendations(prefs,"8888")

#匹配电影——基于物品的过滤
def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
        #人物对换
            result[item][person]=prefs[person][item]
    return result
moviedata=transformPrefs(prefs)
#tuijian=getRecommendations(prefs,'1')
#字典按键排序
def sortP(prefs):
    keys=prefs.keys()
    keys.sort()
    keys.reverse()
    return [prefs[key] for key in keys]
#sp=sorted(tuijian)
#sp2=reversed(sp)
#print(tuijian)
'''
#构建物品（电影）推荐的数据集
def movieSim(prefs,n=1500):
    result={} #字典
    #以物品为中心对偏好矩阵实行倒置处理
    itemP=transformPrefs(prefs)
    c=0
    for item in itemP:
        #针对大数据更新状态变量
        c+=1
        if c%100==0: print ("%d / %d" % (c,len(itemP)))
        #寻找最相识的电影
        scores=topMatches(itemP,item,n=n,similarity=CosSim)
        result[item]=scores
    return result

data1 = movieSim(prefs)
out = open('C:/Users/computer/Desktop/biyesheji/程序/datacos.pkl','wb')
pickle.dump(data1,out)
out.close()
print(data1)
'''
def getRecommendedItems(prefs,itemmatch,user):
    userRatings=prefs[user]
    scores={}
    totalSim={}
    for(item,rating) in userRatings.items():
        for(similarity,item2) in itemmatch[item]:
            if item2 in userRatings:continue
            if similarity>=0.5:
                scores.setdefault(item2, 0)
                scores[item2] += similarity * rating
                totalSim.setdefault(item2, 0)
                totalSim[item2] += similarity
    rankings=[(scores/totalSim[item],item)for item,scores in scores.items()]
    rankings.sort()
    rankings.reverse()
    return rankings
#print(movieSim(prefs))
#wupin=getRecommendedItems(prefs,movieSim(prefs),'1')[0:10]
#print(getRecommendedItems(prefs,movieSim(prefs),'1'))
#print(wupin)
#print(getRecommendedItems(prefs,data2,"1"))
#混合推荐

def itemtc(prefs,itemmatch,user):
    userRatings = prefs[user]
    userRatings2 = [(scores, item) for item, scores in userRatings.items()]
    scores = {}
    totalSim = {}
    for (item, rating) in userRatings.items():
        for (similarity, item2) in itemmatch[item]:
            #print(similarity)
            if item2 in userRatings: continue
            if similarity>=0.3:
                scores.setdefault(item2, 0)
                scores[item2] += similarity * rating
                totalSim.setdefault(item2, 0)
                totalSim[item2] += similarity
    rankings = [(scores / totalSim[item], item) for item, scores in scores.items()]
    rankings.extend(userRatings2)
    rankings.sort()
    rankings.reverse()
    dir = {}
    for i,j in rankings:
        dir.setdefault(j, 0)
        dir[j]=i
    return dir
def hunhe(prefs,itemmatch,user1,similarity=sim_pearson):
    p1 = itemtc(prefs,itemmatch,user1)
    p2 = getRecommendations(prefs,user1)
    totals = {}
    simSums = {}
    p3={}
    p3.update(p1)
    p3.update(p2)
    for item in p1:
        if item in p2:
            p3[item] = (p1[item]+p2[item])/2
    rankings = [(scores, item) for item, scores in p3.items()]
    rankings.sort()
    rankings.reverse()
    return rankings

#print(hunhe(prefs,data2,'1'))
#print(prefs['1'])
#print(hunhe(prefs,data2,'1'))
#输出字典的前10个
def ten(prefs):
    ten1=[]
    count = 0
    for i in prefs:
        ten1.append(i)
        count += 1
        if count == 10:
            break
    return ten1

#itemSim=movieSim(prefs)
#print(movieSim(prefs,n=50))
#print(sim_person(prefs,'196','63'))
#print(topMatches(prefs,'196',n=3))
#print (loadMovieLens()['1'])
#print (getRecommendations(prefs,'1'))
#print(topMatches(prefs,'196',n=5))
#print (getRecommendations(prefs,'196')[0:5])
#print (prefs,itemSim,'186')
#print (getRecommendations(itemSim,'186')[0:10])
#print(getRecommendations(prefs,movieSim(prefs,n=50),'186')[0:30])
#print(ten(sp2))  #输出前10部用户1的推荐电影-基于用户的过滤
#print(ten(wupin)) #输出前10部用户1的推荐电影-基于物品的过滤
#print(prefs)
# a=0
# for key in prefs:
#     a=a+1
# print(a)
#p = itemtc(prefs, data2, '1')
#print(sim_pearson(prefs,'1','222',p))
#print(sim_person(prefs,'1','222'))
#print(prefs['2'])
#print(moviedata)
#a=hunhe(prefs, data2, '8888')
#p=itemtc(prefs,data2,'8888')
#print(a)
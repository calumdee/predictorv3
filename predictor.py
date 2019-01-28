import psycopg2
import matplotlib.pyplot as plt   
from scipy import stats
import numpy as np
import sklearn

def resultToScore(res,hxg,axg):
    scores =[]
    for k in range(0,len(res)):
        x = float(res[k])
        i = 0
        j = 0
        mx = 0
        if x==3:
            score = '(1,0)'
        if x==1:
            score = '(0,0)'
        if x==0:
            score = '(0,1)'
        while i<11:
            j=0
            while j<11:
                if (x==3 and(i>j))or(x==1 and(i==j))or(x==0 and(i<j)):
                    
                    h = stats.distributions.poisson.pmf(i,(hxg[k][0]+axg[k][1])/2)
                    a = stats.distributions.poisson.pmf(j,(axg[k][0]+hxg[k][1])/2)
                    p=h*a
                    
                    if p>mx:
                        mx=p
                        score='('+str(i)+','+str(j)+')'
                j+=1
            i+=1
        scores.append(score)
    return scores
            
            

def predict(season,week,mode,ml):
    conn = psycopg2.connect(config)
    cur = conn.cursor()
    cur.execute(" with t as (select season,gameweek,gameno,tname,team,position,goal,points from games where gameweek < %s and season=%s ), d as (select t.gameweek,t.gameno,t.team,t.tname,l.form as lf,tot.form as tf,t.position as pos, l.txS, lo.txc, t.goal as g,t.points as p from t, (select g,tname,sum(points) as form,avg(xg) as txS from (select t.gameweek as g,ga.gameweek,ga.gameno,ga.tname,ga.points,ga.xg,row_number() over(partition by t.gameweek,ga.tname order by ga.gameweek desc,ga.gameno desc ) as rn from games ga,t where ga.gameweek<t.gameweek and ga.tname=t.tname and ga.team=t.team and ga.season=t.season) as a where rn<=5 group by g,tname) as l, (select g,tname,avg(xg) as txc from (select t.gameweek as g,ga.gameweek,ga.gameno,gb.tname,ga.xg,row_number() over(partition by t.gameweek,gb.tname order by ga.gameweek desc,ga.gameno desc ) as rn from games ga,games gb, t where gb.gameweek<t.gameweek and gb.tname=t.tname and gb.team=t.team and ga.gameweek=gb.gameweek and ga.gameno=gb.gameno and ga.team!=gb.team and ga.season=gb.season and t.season=gb.season) as a where rn<=5 group by g,tname) as lo, (select g,tname,sum(points) as form from (select t.gameweek as g,ga.gameweek,ga.gameno,ga.tname,ga.points,row_number() over(partition by t.gameweek,ga.tname order by ga.gameweek desc,ga.gameno desc ) as rn from games ga,t where ga.gameweek<t.gameweek and ga.tname=t.tname and ga.season=t.season) as a where rn<=5 group by g,tname) as tot where t.tname=l.tname and t.gameweek = l.g and t.tname=tot.tname and t.gameweek=tot.g and t.tname=lo.tname and t.gameweek=lo.g order by gameno,team desc) select h.gameweek,h.gameno, h.tname as hn, a.tname as an, h.lf as hlf, h.tf as htf,h.pos as hpos,h.txs as htxs,h.txc as htxc,a.lf as alf, a.tf as atf,a.pos as apos,a.txs as atxs,a.txc as atxc, (h.g,a.g) as sRes,h.p as oRes from d as h, d as a where h.gameweek=a.gameweek and h.gameno=a.gameno and h.team='H'and a.team='A' order by gameweek,gameno",(week,season,))
    T_train=cur.fetchall()
    cur.execute("with t as (select season,gameweek,gameno,tname,team,position,goal,points from games where gameweek = %s and season=%s ),d as (select t.gameno,t.team,t.tname,l.form as lf,tot.form as tf,t.position as pos, l.txS, lo.txc, t.goal as g,t.points as p from t,(select tname,sum(points) as form,avg(xg) as txS from (select ga.gameweek,ga.gameno,ga.tname,ga.points,ga.xg,row_number() over(partition by ga.tname order by ga.gameweek desc,ga.gameno desc ) as rn from games ga,t  where ga.gameweek<t.gameweek and ga.tname=t.tname and ga.team=t.team and ga.season=t.season) as a where rn<=5 group by tname) as l,(select tname,avg(xg) as txc from (select ga.gameweek,ga.gameno,gb.tname,ga.xg,row_number() over(partition by gb.tname order by ga.gameweek desc,ga.gameno desc ) as rn from games ga,games gb, t  where gb.gameweek<t.gameweek and gb.tname=t.tname and gb.team=t.team and ga.gameweek=gb.gameweek and ga.gameno=gb.gameno and ga.team!=gb.team and ga.season=gb.season and gb.season=t.season) as a where rn<=5 group by tname) as lo,(select tname,sum(points) as form from (select ga.gameweek,ga.gameno,ga.tname,ga.points,row_number() over(partition by ga.tname order by ga.gameweek desc,ga.gameno desc ) as rn from games ga,t  where ga.gameweek<t.gameweek and ga.tname=t.tname and ga.season=t.season) as a where rn<=5 group by tname) as tot where t.tname=l.tname and t.tname=tot.tname and t.tname=lo.tname order by gameno,team desc) , glist as (select a.gameno,a.tname as h,b.tname as a,(a.goal,b.goal) as sRes,a.points as pres from games a,games b where a.gameweek=b.gameweek and a.gameno=b.gameno and a.team='H' and b.team='A' and a.gameweek=%s and a.season=b.season and a.season=%s)select glist.gameno, glist.h,glist.a,coalesce(a.hlf,0),coalesce(a.htf,0),coalesce(a.hpos,0),coalesce(a.htxs,0),coalesce(a.htxc,0),coalesce(a.alf,0),coalesce( a.atf,0),coalesce(a.apos,0),coalesce(a.atxs,0),coalesce(a.atxc,0),glist.sRes,glist.pres from glist left join(select h.gameno, h.tname as hn, a.tname as an, h.lf as hlf, h.tf as htf,h.pos as hpos,h.txs as htxs,h.txc as htxc,a.lf as alf, a.tf as atf,a.pos as apos,a.txs as atxs,a.txc as atxc from d as h, d as a where h.gameno=a.gameno and h.team='H'and a.team='A' order by gameno) as a on glist.gameno=a.gameno and glist.h=a.hn and glist.a=a.an order by glist.gameno",(week,season,week,season,))
    T_test=cur.fetchall()    
    X_train=np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
    m = np.array(T_test)[:,0:3]
    
    if not(T_train==[]):
        X_train = np.append(X_train,np.array(T_train)[:,4:14].astype(float),axis=0)
    X_test = np.array(T_test)[:,3:13].astype(float)
    
    if mode == 'score':
        classes =['(0,0)','(0,1)','(0,2)','(0,3)','(0,4)','(0,5)','(0,6)','(0,7)','(0,8)','(0,9)','(1,0)','(1,1)','(1,2)','(1,3)','(1,4)','(1,5)','(1,6)','(1,7)','(1,8)','(1,9)','(2,0)','(2,1)','(2,2)','(2,3)','(2,4)','(2,5)','(2,6)','(2,7)','(2,8)','(2,9)','(3,0)','(3,1)','(3,2)','(3,3)','(3,4)','(3,5)','(3,6)','(3,7)','(3,8)','(3,9)','(4,0)','(4,1)','(4,2)','(4,3)','(4,4)','(4,5)','(4,6)','(4,7)','(4,8)','(4,9)','(5,0)','(5,1)','(5,2)','(5,3)','(5,4)','(5,5)','(5,6)','(5,7)','(5,8)','(5,9)','(6,0)','(6,1)','(6,2)','(6,3)','(6,4)','(6,5)','(6,6)','(6,7)','(6,8)','(6,9)','(7,0)','(7,1)','(7,2)','(7,3)','(7,4)','(7,5)','(7,6)','(7,7)','(7,8)','(7,9)','(8,0)','(8,1)','(8,2)','(8,3)','(8,4)','(8,5)','(8,6)','(8,7)','(8,8)','(8,9)','(9,0)','(9,1)','(9,2)','(9,3)','(9,4)','(9,5)','(9,6)','(9,7)','(9,8)','(9,9)']
        y_train=np.array([['(1,0)'],['(1,1)'],['(0,1)']])
        if not(T_train==[]):
            y_train = np.append(y_train,np.array(T_train)[:,14:15],axis=0)
    y_test = np.array(T_test)[:,13:14]
    if mode == 'point':
        classes=[3,1,0]
        y_train=np.array([[3],[1],[0]])
        if not(T_train==[]):
            y_train = np.append(y_train,np.array(T_train)[:,15:16].astype(float),axis=0)
        #y_test = np.array(T_test)[:,14:15].astype(float)
    cur.close()

    if conn is not None:
        conn.close()
        
    if ml == 'bayes':
        nbc = NBC(feature_types=['r','r','r','r','r','r','r','r','r','r'], num_classes=3, class_name=classes)
        nbc.fit(X_train,y_train)
        y_hat=nbc.predict(X_test)
    
    if ml =='lr':
        lr = LogisticRegression()
        lr.fit(X_train,y_train.astype(str))
        y_hat = lr.predict(X_test)
    
    if mode == 'point':
        
        y_hat = resultToScore(y_hat,X_test[:,3:5],X_test[:,8:10])
            
    
    
    return m,y_test,y_hat

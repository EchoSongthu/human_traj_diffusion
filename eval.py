import setproctitle
import scipy.stats
import numpy as np
from collections import Counter
from math import radians, cos, sin, asin, sqrt

import pdb
import pickle

import setproctitle
proc_title = "evaluation"
setproctitle.setproctitle(proc_title)

'''
evalute 评价jsd
连续呆在一个地方，指标都计算一次
'''

leng = 168

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 
    distance=round(distance/1000,3)
    return distance


class EvalUtils(object):
    """
    some commonly-used evaluation tools and functions
    """

    @staticmethod
    def filter_zero(arr):
        """
        remove zero values from an array
        :param arr: np.array, input array
        :return: np.array, output array
        """
        arr = np.array(arr)
        filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
        return filtered_arr

    @staticmethod
    def arr_to_distribution(arr, min, max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max 区间个数
        :return: np.array, output distribution array
        """
        distribution, base = np.histogram(
            arr, np.arange(
                min, max, float(
                    max - min) / bins))
        return distribution, base[:-1]

    @staticmethod
    def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-14)
        p2 = p2 / (p2.sum()+1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + \
            0.5 * scipy.stats.entropy(p2, m)
        return js
    
   
class IndividualEval(object):

    def __init__(self):
        self.max_locs = 32400
        self.max_distance = 252.007
        
        max_grid = 180
        GRID_SIZE = 1000
        lon_l, lon_r, lat_b, lat_u = 115.43, 117.52, 39.44, 41.05 # Beijing
        earth_radius = 6378137.0
        pi = 3.1415926535897932384626
        meter_per_degree = earth_radius * pi / 180.0
        lat_step = GRID_SIZE * (1.0 / meter_per_degree)
        ratio = np.cos((lat_b + lat_u) * np.pi / 360)
        lon_step = lat_step / ratio

        self.X = []
        self.Y = []
        for grid_id in range(1,self.max_locs+1):
            x = grid_id // max_grid #经度
            y = grid_id - x*max_grid - 1 #纬度
            self.X.append((x+0.5)*lon_step + lon_l)
            self.Y.append((y+0.5)*lat_step + lat_b)
        
    def get_topk_visits(self,trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        for traj in trajs:
            traj_temp = []
            traj_temp.append(traj[0])
            for id_,t in enumerate(traj[1:]):
                if t != traj[id_]:
                    traj_temp.append(t)
            topk = Counter(traj_temp).most_common(k)
            for i in range(len(topk), k):
                # supplement with (loc=-1, freq=0)
                topk += [(-1, 0)]
            loc = [l for l, _ in topk]
            freq = [f for _, f in topk]
            loc = np.array(loc, dtype=int)
            freq = np.array(freq, dtype=float) / len(traj_temp)
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)
        topk_visits_loc = np.array(topk_visits_loc, dtype=int)
        topk_visits_freq = np.array(topk_visits_freq, dtype=float)
        return topk_visits_loc, topk_visits_freq

    def get_overall_topk_visits_freq(self, trajs, k):
        _, topk_visits_freq = self.get_topk_visits(trajs, k)
        mn = np.mean(topk_visits_freq, axis=0)
        return mn / np.sum(mn)

    def get_distances(self, trajs):
        '''
            cumulative travel distance of per user (normalized by hour)
            对每个用户计算一个累积distance，对序列长度（总时长）做归一化
        '''
        distances = []
        seq_len = leng
        for traj in trajs:
            dis = 0.
            for i in range(seq_len - 1):
                dis += geodistance(self.X[traj[i]],self.Y[traj[i]],self.X[traj[i + 1]],self.Y[traj[i + 1]])
            distances.append(dis/leng)
        distances = np.array(distances, dtype=float)
        return distances
    
    '''
    def get_home(self,trajs):
        homes = []
        for traj in trajs:
            traj_merge = merge(traj)
            home = identify_home(traj_merge)
            homes.append(home)
        return homes
    
    def get_home_time(self,trajs,homes):
        t = []
        for id_,traj in enumerate(trajs):
            time = 0
            home = homes[id_]
            for i in traj:
                if i == home:
                    time += 1
            t.append(time)
        return np.array(t)/leng 

    def get_home_days(self,trajs,homes):
        d = []
        for id_,traj in enumerate(trajs):
            day = 0
            home = homes[id_]
            for i in range(7):
                for j in range(i*24,(i+1)*24):
                    if traj[j] == home:
                        day += 1
                        break
            d.append(day)
        return np.array(d)/7

    def get_home_travels(self,trajs,homes):
        travels = []
        for id_,traj in enumerate(trajs):
            travel = 0
            home = homes[id_]
            for id_,i in enumerate(traj[1:]):
                if traj[id_] == home and i != home:
                    travel += 1
            travels.append(travel)
        return np.array(travels)/leng
    '''

    def get_durations(self, trajs):
        d = []
        for traj in trajs:
            num = 1
            for i, lc in enumerate(traj[1:]):
                if lc == traj[i]:
                    num += 1
                else:
                    d.append(num)
                    num = 1
        return np.array(d)/leng
    
    def get_periodicity(self, trajs):
        """
        stat how many repetitions within a single trajectory
        :param trajs:
        :return:
        """
        reps = []
        for traj in trajs:
            reps.append(float(len(set(traj)))/leng)
        reps = np.array(reps, dtype=float)
        return reps

    def get_geogradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        for traj in trajs:
            traj_temp = [] # 连续访问去重
            traj_temp.append(traj[0])
            for id_,t in enumerate(traj[1:]):
                if t != traj[id_]:
                    traj_temp.append(t)
            
            xs = np.array([self.X[t] for t in traj_temp])
            ys = np.array([self.Y[t] for t in traj_temp])
            lng1, lat1 = np.mean(xs), np.mean(ys)
            rad = []
            for i in range(len(xs)):         
                lng2 = xs[i]
                lat2 = ys[i]
                distance = geodistance(lng1,lat1,lng2,lat2)
                rad.append(distance)
            rad = np.mean(np.array(rad, dtype=float))

            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius

    def get_individual_jsds(self, t1, t2):
        """
        get jsd scores of individual evaluation metrics
        :param t1: test_data
        :param t2: gene_data
        :return:
        """
        t1 = t1-1
        t2 = t2-1

        # travel distance
        d1 = self.get_distances(t1)
        d2 = self.get_distances(t2)

        d1_dist, _ = EvalUtils.arr_to_distribution(
            d1, 0, self.max_distance, 10000)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            d2, 0, self.max_distance, 10000)
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)
        

        # gyration radius
        g1 = self.get_geogradius(t1)
        g2 = self.get_geogradius(t2)

        # max_d = max(g1[g1.argmax()],g2[g2.argmax()])
        g1_dist, _ = EvalUtils.arr_to_distribution(
            g1, 0, self.max_distance, 10000) # 上界？TODO
        g2_dist, _ = EvalUtils.arr_to_distribution(
            g2, 0, self.max_distance, 10000)
        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)
        

        # stay duration
        du1 = self.get_durations(t1)
        du2 = self.get_durations(t2)     
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, leng)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, leng)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)
        

        # daily loc
        p1 = self.get_periodicity(t1)
        p2 = self.get_periodicity(t2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, leng)
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, leng)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)

        
        l1 =  CollectiveEval.get_visits(t1,self.max_locs)
        l2 =  CollectiveEval.get_visits(t2,self.max_locs)
        l1_dist, _ = CollectiveEval.get_topk_visits(l1, 100)
        l2_dist, _ = CollectiveEval.get_topk_visits(l2, 100)
        l1_dist, _ = EvalUtils.arr_to_distribution(l1_dist,0,1,500)
        l2_dist, _ = EvalUtils.arr_to_distribution(l2_dist,0,1,500)
        l_jsd = EvalUtils.get_js_divergence(l1_dist, l2_dist)


        f1 = self.get_overall_topk_visits_freq(t1, 100)
        f2 = self.get_overall_topk_visits_freq(t2, 100)
        f1_dist, _ = EvalUtils.arr_to_distribution(f1,0,1,500) # TODO
        f2_dist, _ = EvalUtils.arr_to_distribution(f2,0,1,500)
        f_jsd = EvalUtils.get_js_divergence(f1_dist, f2_dist)


        '''
        home_test = self.get_home(t1) #计算每个用户的home
        home_gen = self.get_home(t2)

        j1 = self.get_home_time(t1,home_test)
        j2 = self.get_home_time(t2,home_gen)
        j1_dist, _ = EvalUtils.arr_to_distribution(j1,0,1,500)
        j2_dist, _ = EvalUtils.arr_to_distribution(j2,0,1,500)
        j_jsd = EvalUtils.get_js_divergence(j1_dist, j2_dist)

        da1 = self.get_home_days(t1,home_test)
        da2 = self.get_home_days(t2,home_gen)
        da1_dist, _ = EvalUtils.arr_to_distribution(da1,0,1,500)
        da2_dist, _ = EvalUtils.arr_to_distribution(da2,0,1,500)
        da_jsd = EvalUtils.get_js_divergence(da1_dist, da2_dist)

        tr1 = self.get_home_travels(t1,home_test)
        tr2 = self.get_home_travels(t2,home_gen)
        tr1_dist, _ = EvalUtils.arr_to_distribution(tr1,0,1,500)
        tr2_dist, _ = EvalUtils.arr_to_distribution(tr2,0,1,500)
        tr_jsd = EvalUtils.get_js_divergence(tr1_dist, tr2_dist)
        '''

        
        return d_jsd,  g_jsd,  du_jsd,  p_jsd, l_jsd, f_jsd #, j_jsd , da_jsd, tr_jsd


class CollectiveEval(object):
    """
    collective evaluation metrics
    """
    @staticmethod
    def get_visits(trajs,max_locs):
        """
        get probability distribution of visiting all locations
        :param trajs:
        :return:
        """
        visits = np.zeros(shape=(max_locs), dtype=float)
        for traj in trajs:
            last = traj[0]
            visits[last] += 1
            for t in traj[1:]:
                if t != last:
                    visits[t] += 1
                    last = t
        visits = visits / np.sum(visits)
        return visits
    # 连续呆在一个地方：算一次

    @staticmethod
    def get_topk_visits(visits, K):
        """
        get top-k visits and the corresponding locations
        :param trajs:
        :param K:
        :return:
        """
        locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
        locs_visits.sort(reverse=True, key=lambda d: d[1])
        topk_locs = [locs_visits[i][0] for i in range(K)]
        topk_probs = [locs_visits[i][1] for i in range(K)]
        return np.array(topk_probs), topk_locs


def evaluate():
    individualEval = IndividualEval()

    is_model = False
    # dataset = 'chinamobile'
    dataset = 'tencent'
    exp_name = '57'
    print("dataset=",dataset)

    test_data_ = [pickle.load(open(f'../dataset/{dataset}/split/test_data_train.pkl', 'rb')),
                pickle.load(open(f'../dataset/{dataset}/split/test_data_val.pkl', 'rb'))]

    for test_data in test_data_:
        if is_model:
            for num in range(160000, 170000, 10000):
                gen_path =f"./improved-diffusion/genout/exp{exp_name}/gen_data_step_{num}.pkl"
                gene_data = pickle.load(open(gen_path, 'rb'))
                print(individualEval.get_individual_jsds(test_data,gene_data))
            print("model evaluation: well done")
        else:
            # gen_path =f"../../dataset/{dataset}/timegeo.pkl"
            gen_path = "/data/zmy/human_traj_diffusion/improved-diffusion/genout_control/home_control_tencent.pkl"
            print(gen_path)
            gene_data = pickle.load(open(gen_path, 'rb'))
            print(individualEval.get_individual_jsds(test_data,gene_data))
            print("baseline evaluation: well done")
        print("==========================")


if __name__ == "__main__":
    evaluate()
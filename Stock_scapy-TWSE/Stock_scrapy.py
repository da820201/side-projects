import requests
import random
import pandas as pd
import time
import datetime
from bs4 import BeautifulSoup as bs
import multiprocessing as mp
import threading as td
import psutil
import os
import socket
from db_ORM import DB_modify
import json
import traceback
import sys
import queue as qe


def get_host_ip():
    ip = None

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


class free_ip(object):
    defults_headers = {
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Safari/535.19",
    }

    defults_params = {
        "random_UA": True,
        "kuaidaili_IP": [],
        "xicidaili_IP": [],
        "ip3366_IP": []
    }

    defults_free_ip_web = {
        "kuaidaili": 'http://www.kuaidaili.com/proxylist/',
        "xicidaili": 'https://www.xicidaili.com/wn/',
        "ip3366": 'http://www.ip3366.net/free/?stype=1&page=',
        "spys_one": 'http://spys.one/free-proxy-list/TW/',
        "proxyranker": "https://proxyranker.com/taiwan"
    }

    _stock_dict = {
        '麗正': '2302',
        '聯電': '2303',
        '華泰': '2329',
        '台積電': '2330',
        '旺宏': '2337',
        '光罩': '2338',
        '茂矽': '2342',
        '華邦電': '2344',
        '順德': '2351',
        '矽統': '2363',
        '菱生': '2369',
        '瑞昱': '2379',
        '威盛': '2388',
        '凌陽': '2401',
        '南亞科': '2408',
        '統懋': '2434',
        '偉詮電': '2436',
        '超豐': '2441',
        '京元電子': '2449',
        '創見': '2451',
        '聯發科': '2454',
        '義隆': '2458',
        '強茂': '2481',
        '晶豪科': '3006',
        '聯陽': '3014',
        '嘉晶': '3016',
        '聯詠': '3034',
        '智原': '3035',
        '揚智': '3041',
        '立萬利': '3054',
        '聯傑': '3094',
        '景碩': '3189',
        '虹冠電': '3257',
        '京鼎': '3413',
        '創意': '3443',
        '晶相光': '3530',
        '台勝科': '3532',
        '誠創': '3536',
        '敦泰': '3545',
        '辛耘': '3583',
        '通嘉': '3588',
        '世芯-KY': '3661',
        '達能': '3686',
        '日月光投控': '3711',
        '新唐': '4919',
        '凌通': '4952',
        '天鈺': '4961',
        '十銓': '4967',
        '立積': '4968',
        '祥碩': '5269',

    }
    _df = pd.DataFrame(_stock_dict.values(), _stock_dict.keys(), ['stock_id'])
    _fd = pd.DataFrame(_stock_dict.keys(), _stock_dict.values(), ['stock_name'])

    @classmethod
    def run(cls, n):
        if n in cls.defults_params:
            return cls.defults_params[n]
        else:
            return "unrecognized params {}".format(n)

    def __init__(self, **kwargs):
        self.db = DB_modify()
        self.total_error = 0
        self.__dict__.update(self.defults_params)
        self.__dict__.update(kwargs['kwargs']) if "kwargs" in kwargs else self.__dict__.update(kwargs)
        self.defults_headers.update(
            {"User-Agent": self.user_agent_list() if self.random_UA else self.defults_headers['User-Agent']})

    def log_writer(self, data: list or None or dict or str):
        f = open('./log.txt', newline='', mode='w', encoding='utf-8')
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, dict):
            pass
        elif data is None:
            return True
        elif not isinstance(data, list):
            raise TypeError("Variabel data should be dict or str or list")

        for i in data:
            f.write(i)
            f.write('\n')

    def user_agent_list(self):
        UA_list = [
            "Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Safari/535.19",
            "Mozilla/5.0 (Linux; U; Android 4.0.4; en-gb; GT-I9300 Build/IMM76D) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30",
            "Mozilla/5.0 (Linux; U; Android 2.2; en-gb; GT-P1000 Build/FROYO) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
            "Mozilla/5.0 (Windows NT 6.2; WOW64; rv:21.0) Gecko/20100101 Firefox/21.0",
            "Mozilla/5.0 (Android; Mobile; rv:14.0) Gecko/14.0 Firefox/14.0",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36"
        ]
        result = random.choice(UA_list)
        return result

    def script(self, target: list or str):
        if not isinstance(target, list):
            target = [target]

        job = {}
        multip = {}
        for i, m in enumerate(target):
            print("Job: {} is starting..".format(m))
            job.update({"q{}".format(i): mp.Queue()})
            arg = (m, job["q{}".format(i)],)
            multip.update({"l{}".format(i): mp.Process(target=self.free_ip_get, args=arg)})
            multip["l{}".format(i)].start()
            time.sleep(0.05)

        for i in range(0, len(target)):
            multip["l{}".format(i)].join()

        result = []
        for it in job.keys():
            if job[it].qsize() != 0:
                result = result + job[it].get()[0]

        result = list(set(result))
        result = self.test_ip(result)

        return result

    def twse_json_worker(self, result):
        return json.loads(result)

    def yahoo_json_worker(self, result):
        now = datetime.datetime.utcnow().replace(microsecond=0)
        if '"143":0' in result:
            # 0900 ~ 0959要先去除"0"才能進json.loads()
            response = result.split('"143":0')[0] + '"143":' + result.split('"143":0')[1]
            response = json.loads(response[5:-2])
            time_str = str(response['mem']['144']) + '0' + str(response['mem']['143'])
            timestamp = datetime.datetime.strptime(time_str, "%Y%m%d%H%M%S") - datetime.timedelta(hours=8)
        else:
            response = json.loads(result[5:-2])
            if response['mem'].get('144', False) and response['mem'].get('143', False):
                time_str = str(response['mem']['144']) + str(response['mem']['143'])
                timestamp = datetime.datetime.strptime(time_str, "%Y%m%d%H%M%S") - datetime.timedelta(hours=8)
            else:
                timestamp = now
        if timestamp < now - datetime.timedelta(seconds=5):
            # 如果timestamp 比現在時間早5秒以上，就將當盤成交量設為0
            response['mem']['413'] = 0
            timestamp = now
        if "mem" in response.keys():
            response['mem'].update({'timestamp': timestamp})
            result = response['mem']
        return result

    def requests_Er_url(self, Er_q_data: dict, ip: str, IP_Q, Er_q, command_code) -> None:
        target_id = Er_q_data['id']
        url = Er_q_data['url']
        belong_qe = Er_q_data['qe']
        Error_times = Er_q_data['Er_times']
        now_time = Er_q_data['now_time']

        if Error_times > 10:
            print("back up: This proxy {} that Erred more than 5 times, system switch proxy".format(ip))
            if command_code.qsize() > 0:
                pass
            else:
                IP_Q.put(ip)
        else:
            try:
                if "twse" in url: url = "{}{}".format(url, now_time)

                if ip is None:
                    data = requests.get(url=url)
                else:
                    data = requests.get(url=url, proxies={'https//': ip})

                if data.ok:
                    if "twse" in url:
                        result = data.json()
                    else:
                        result = self.yahoo_json_worker(data.text)
                    # print("use back up: {}".format(result))
                    belong_qe.put(result)
                    if command_code.qsize() > 0:
                        pass
                    else:
                        IP_Q.put(ip)
                else:
                    print("Erred target: {},Proxy: {},time: {}".format(target_id, ip, now_time))
                    Er_q_data['Er_times'] += 1
                    Er_q.put(Er_q_data)
                    if command_code.qsize() > 0:
                        pass
                    else:
                        IP_Q.put(ip)
            except:
                print("Erred target: {},Proxy: {},time: {}".format(target_id, ip, now_time))
                Er_q_data['Er_times'] += 1
                Er_q.put(Er_q_data)
                if command_code.qsize() > 0:
                    pass
                else:
                    IP_Q.put(ip)

    def requests_Ev_url(self, Ep_q_data: dict, ip: str or None, Er_q, E_IP_Q, command_code):
        target_id = Ep_q_data['id']
        url = Ep_q_data['url']
        rqe = Ep_q_data['qe']
        Ep_rror_times = 0
        while True:
            if Ep_rror_times > 5 and E_IP_Q.qsize() > 0:
                print("work:This proxy {} that Erred more than 5 times, system switch proxy".format(ip))
                if command_code.qsize() > 0:
                    pass
                else:
                    Ep_rror_times = 0
                    ip = E_IP_Q.get()
            else:
                now_time = int(time.time())
                try:
                    if "twse" in url:
                        url = "{}{}".format(url, now_time)

                    if ip is None:
                        data = requests.get(url=url)
                    else:
                        data = requests.get(url=url, proxies={'https//': ip})

                    if data.ok:
                        if "twse" in url:
                            result = data.json()
                        else:
                            result = self.yahoo_json_worker(data.text)
                        # print("work:{}".format(result))
                        rqe.put(result)
                    else:
                        print("Erred target: {},Proxy: {},time: {}".format(target_id, ip, now_time))
                        Er_q_dict = Ep_q_data
                        Er_q_dict['Er_times'] += 1
                        Er_q_dict['now_time'] = now_time
                        Er_q.put(Er_q_dict)
                        Ep_rror_times += 1
                except Exception as Er:
                    # print(Er)
                    print("Erred target: {},Proxy: {},time: {}".format(target_id, ip, now_time))
                    Er_q_dict = Ep_q_data
                    Er_q_dict['Er_times'] += 1
                    Er_q_dict['now_time'] = now_time
                    Er_q.put(Er_q_dict)
                    Ep_rror_times += 1

                time.sleep(5)

    def ip_getter(self, er_ip_q, ev_ip_q, command_code, PL, PID_q):
        PID = os.getpid()
        PID_q.put(PID)

        def empty_q(qe):
            while True:
                if qe.qsize() > 0:
                    qe.get()
                else:
                    break

        while True:
            time.sleep(30)
            ip = self.script(["spys_one", 'proxyranker'])
            command_code.put(1)
            empty_q(er_ip_q)
            empty_q(ev_ip_q)

            i = 0
            len_ip = len(ip.keys())
            for it in ip.keys():
                if i > int(len_ip * PL):
                    ev_ip_q.put(it)
                else:
                    er_ip_q.put(it)
                i += 1
            command_code.get()

    def listener(self, Er_q, IP_Q, command_code, PID_q):
        PID = os.getpid()
        PID_q.put(PID)

        while True:
            if Er_q.qsize() > 0:
                if command_code.qsize() > 0:
                    time.sleep(0.5)
                else:
                    Er_q_args = (Er_q.get(), IP_Q.get(), IP_Q, Er_q, command_code,)
                    Er_q_process = td.Thread(target=self.requests_Er_url, args=Er_q_args)
                    Er_q_process.start()
            else:
                time.sleep(0.5)

    def eventer(self, target: list, IP_Q, Er_q, carry_: int, main_q, command_code, PID_q):
        Ep_data_form = {
            "id": "",
            "url": "",
            "qe": "",
            "now_time": "",
            "Er_times": 0,
        }

        target_dict = {}

        PID = os.getpid()
        PID_q.put(PID)
        now_ip = None

        for it, mt in enumerate(target):
            if (it + 1) % carry_ == 0 or now_ip is None:
                now_ip = IP_Q.get()
            Ep_dict = Ep_data_form
            Ep_dict.update({"id": mt[1],
                            "qe": mt[2],
                            "url": mt[0]})
            main_q.put({mt[1]: mt[2]})
            args = (Ep_dict, now_ip, Er_q, IP_Q, command_code,)
            # Ep_mp = mp.Process(target=self.requests_Ev_url, args=args)
            Ep_mp = td.Thread(target=self.requests_Ev_url, args=args)
            target_dict.update({"target_{}".format(it): Ep_mp})
            Ep_mp.start()

        print("eventer done")

        while True:
            for it, mt in enumerate(target_dict.keys()):
                target_is_alive = target_dict[mt].isAlive()
                if target_is_alive is False:
                    if IP_Q.qsize() > 0:
                        now_ip = IP_Q.get()
                    else:
                        now_ip = None
                    Ep_dict = Ep_data_form
                    Ep_dict.update({"id": mt[1],
                                    "qe": mt[2],
                                    "url": mt[0]})
                    main_q.put({mt[1]: mt[2]})
                    args = (Ep_dict, now_ip, Er_q, IP_Q, command_code,)
                    Ep_mp = td.Thread(target=self.requests_Ev_url, args=args)
                    target_dict.update({"target_{}".format(it): Ep_mp})
                    Ep_mp.start()
            time.sleep(0.1)

    def except_logger(self, e, result=None):
        self.total_error += 1
        cl, exc, tb = sys.exc_info()  # 取得Call Stack
        errMsg = "total_error:{} type:{} value:{} e:{} result:{}".format(self.total_error, cl, exc, e, result)
        print(errMsg)
        self.db.logger(errMsg, str(traceback.extract_tb(tb)))

    def td_data_wirte(self, stock_id, target: mp.Queue, timer_q: mp.Queue, input_q: mp.Queue):
        timer_push_dict = {}
        while True:

            if target.qsize() > 0:
                commit_data = target.get()
                continue

            for i in commit_data:
                if i.qsize() > 0:
                    d = i.get()
                    if d is not None:
                        if "msgArray" in str(d):
                            timer_push_dict.update({"twse": d})
                        else:
                            timer_push_dict.update({"yahoo": d})

            if timer_q.qsize() > 0:
                # print("{}: {}".format(stock_id, timer_q.qsize()))
                input_ = None

                if "twse" in timer_push_dict.keys():
                    input_ = timer_push_dict["twse"]
                elif "yahoo" in timer_push_dict.keys():
                    input_ = timer_push_dict["yahoo"]
                if input_ is not None:
                    try:
                        input_q.put(input_)
                        # print(input_)
                        timer_q.get()
                    except Exception as e:
                        self.except_logger(e, input_)
                    finally:
                        timer_push_dict = {}

            time.sleep(0.05)

    def _TT(self, q: mp.Queue):
        while True:
            time.sleep(5)
            q.put(1)

    def timer(self, timer_queue: mp.Queue, PID_q: mp.Queue, data_input_list_Q: mp.Queue):
        pid = os.getpid()
        PID_q.put(pid)
        tq = mp.Manager().Queue()
        t_arg = (tq,)
        t_td = td.Thread(target=self._TT, args=t_arg)
        t_td.start()

        time_dict = {}
        input_list = []
        while True:
            if timer_queue.qsize() > 0:
                time_dict.update(timer_queue.get())
                continue

            while True:
                if data_input_list_Q.qsize() > 0:
                    input_list.append(data_input_list_Q.get())
                else:
                    break

            if tq.qsize() > 0:
                print(len(input_list))
                self.db.insert_best_5(input_list)
                self.db.commit_all()
                input_list = []
                for i in time_dict.keys():
                    if time_dict[i].qsize() > 0:
                        pass
                    else:
                        time_dict[i].put(1)
                tq.get()

            time.sleep(0.05)

    def data_wirte(self, main_q: mp.Queue, queue_list, PID_queue, q_d_mp, input_Q):
        q_dict = {}
        pid = os.getpid()
        PID_queue.put(pid)

        def data_T(d_dict, dT, id_):
            if isinstance(d_dict[id_], list):
                d_dict[id_].append(dT[id_])
                n_q = list(set(d_dict[id_]))
                d_dict[id_] = n_q
            else:
                n_q = list(set([q_dict[id_], i[id_]]))

            return n_q

        for i in queue_list:
            for it in i.keys():
                name_ = it
                break

            if name_ in q_dict.keys():
                n_q = data_T(q_dict, i, name_)
                q_dict[name_] = n_q
            else:
                q_dict.update({name_: i[name_]})

        while True:
            if main_q.qsize() > 0:
                mq = main_q.get()
                for i in mq:
                    for it in i.keys():
                        name_ = it
                        break

                if name_ in q_dict.keys():
                    n_q = data_T(q_dict, mq, name_)
                    q_dict[name_] = n_q
                else:
                    q_dict.update({name_: mq[name_]})
                continue
            else:
                break

        td_dict = {}
        for i in q_dict.keys():
            td_q = mp.Manager().Queue()
            timer_mp = mp.Manager().Queue()
            q_d_mp.put({i: timer_mp})
            td_q.put(q_dict[i])
            td_dict.update({i: td_q})
            arg = (i, td_dict[i], timer_mp, input_Q,)
            t_td = td.Thread(target=self.td_data_wirte, args=arg)
            t_td.start()

        while True:
            if main_q.qsize() > 0:
                mq = main_q.get()
                for i in mq:
                    for it in i.keys():
                        name_ = it
                        break

                if name_ in q_dict.keys():
                    n_q = data_T(q_dict, mq, name_)
                    if q_dict[name_] != n_q:
                        td_dict[name_].put(mq[name_])
                        q_dict[name_] = n_q
                else:
                    q_dict.update({name_: mq[name_]})
                    td_q = mp.Manager().Queue()
                    td_q.put(q_dict[name_])
                    td_dict.update({name_: td_q})
                    timer_mp = mp.Manager().Queue()
                    q_d_mp.put({name_: timer_mp})
                    arg = (name_, td_dict[name_], timer_mp, input_Q,)
                    t_td = td.Thread(target=self.td_data_wirte, args=arg)
                    t_td.start()

            time.sleep(0.1)

    def scrapy_for_best_bid_and_ask(self, counter, stock_id: list or dict, IP: list or dict, carry_num=3, delay=5):
        stock_target_list = []
        command_code = mp.Manager().Queue()
        main_q = mp.Manager().Queue()
        self_IP = get_host_ip()
        PL = 0.5
        timer_q = mp.Manager().Queue()

        log = self.log_writer(None)
        if log:
            log = self.log_writer
        else:
            raise IOError("can't open log.txt")

        def init_param(self, IP, stock_target_list, carry_num, yahoo=True):
            IP_list = []
            yahoo_ = []
            if isinstance(carry_num, float):
                carry_num = int(carry_num)

            if isinstance(stock_id, dict):
                if yahoo is not True:
                    yahoo_ = None
                for st_it in stock_id.keys():
                    mp_q = mp.Manager().Queue()
                    stock_target_list.append([
                        "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={}.tw&json=1&delay=0&_=".format(
                            stock_id[st_it]), stock_id[st_it][-4:], mp_q])
                    # main_q.put({stock_id[st_it][-4:]: mp_q})
                    if yahoo:
                        yahoo_.append([
                            'https://tw.quote.finance.yahoo.net/quote/q?type=tick&perd=5s&mkt=10&sym={}'.format(
                                stock_id[st_it][-4:]), stock_id[st_it][-4:], mp_q])
                        # print('https://tw.quote.finance.yahoo.net/quote/q?type=tick&perd=5s&mkt=10&sym={}'.format(stock_id[st_it][-4:]))
                        # main_q.put({stock_id[st_it][-4:]: mp_q})
            elif isinstance(stock_id, list):
                if yahoo is not True:
                    yahoo_ = None
                for st_it in stock_id:
                    mp_q = mp.Manager().Queue()
                    stock_target_list.append([
                        "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={}.tw&json=1&delay=0&_=".format(
                            st_it), st_it[-4:], mp_q])
                    # main_q.put({st_it[-4:]: mp_q})
                    if yahoo:
                        yahoo_.append([
                            'https://tw.quote.finance.yahoo.net/quote/q?type=tick&perd=5s&mkt=10&sym={}'.format(
                                st_it[-4:]), st_it[-4:], mp_q])
                        # print('https://tw.quote.finance.yahoo.net/quote/q?type=tick&perd=5s&mkt=10&sym={}'.format(st_it[-4:]))
                        # main_q.put({st_it[-4:]: mp_q})
            else:
                raise TypeError("Variable stock_id should be list or dict")

            if isinstance(IP, dict):
                for IP_it in IP.keys():
                    IP_list.append(IP_it)
            elif IP is None:
                IP = self.script(['spys_one', "proxyranker"])
                for IP_it in IP.keys():
                    IP_list.append(IP_it)
            elif not isinstance(IP, list):
                raise TypeError("Variable IP should be list or dict")
            else:
                IP_list = IP

            return IP_list, stock_target_list, carry_num, yahoo_

        IP_list, stock_target_list, carry_num, stock_target_list_yahoo = init_param(self, IP, stock_target_list,
                                                                                    carry_num)
        len_IP_list = len(IP_list)
        len_target_list = len(stock_target_list)
        PID_queue = mp.Manager().Queue()
        input_Q = mp.Manager().Queue()

        Error_queue = mp.Manager().Queue()
        Er_IP_q = mp.Manager().Queue()
        Er_q_ip = IP_list[int(len_IP_list * 0.5):]
        for eripq in Er_q_ip:
            Er_IP_q.put(eripq)
        Er_q_args = (Error_queue, Er_IP_q, command_code, PID_queue,)
        mp_listener = mp.Process(target=self.listener, args=Er_q_args)
        mp_listener.start()

        Ev_q_ip = IP_list[:int(len_IP_list * 0.5)]
        Ev_IP_q = mp.Manager().Queue()
        for evipq in Ev_q_ip:
            Ev_IP_q.put(evipq)

        for i in range(int(len(stock_target_list) / 10)):
            target = stock_target_list[i * 10:(i + 1) * 10]
            Ev_q_args = (target, Ev_IP_q, Error_queue, carry_num, main_q, command_code, PID_queue,)
            mp_eventer = mp.Process(target=self.eventer, args=Ev_q_args)
            mp_eventer.start()

        print("twse eventer was wake up")

        if stock_target_list_yahoo is not None:
            Ev_q_args = (stock_target_list_yahoo, Ev_IP_q, Error_queue, carry_num, main_q, command_code, PID_queue,)
            mp_eventer_yahoo = mp.Process(target=self.eventer, args=Ev_q_args)
            mp_eventer_yahoo.start()
            print("yahoo eventer was wake up")

        gfip_q_args = (Er_IP_q, Ev_IP_q, command_code, PL, PID_queue,)
        mp_gfip = mp.Process(target=self.ip_getter, args=gfip_q_args)
        mp_gfip.start()

        result_queue = []
        len_T = 0
        while True:
            if stock_target_list_yahoo is not None and len_T == 0:
                len_target_list = len_target_list * 2
                len_T = 1

            if main_q.qsize() > 0:
                try:
                    result_queue.append(main_q.get())
                except:
                    pass
                continue
            elif len_target_list <= len(result_queue):
                break
            else:
                print("wait all target processing created, now length or queue:{}".format(len(result_queue)))
                time.sleep(0.5)

        timer_args = (timer_q, PID_queue, input_Q,)
        timer_mp = mp.Process(target=self.timer, args=timer_args)
        timer_mp.start()

        time.sleep(0.5)
        data_wirte_args = (main_q, result_queue, PID_queue, timer_q, input_Q,)
        mp_data_wirte = mp.Process(target=self.data_wirte, args=data_wirte_args)
        mp_data_wirte.start()

        while True:
            if mp_eventer.is_alive() and mp_gfip.is_alive():
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory()
                print_line = "listener eventer gfip"
                if stock_target_list_yahoo is not None and mp_eventer_yahoo.is_alive():
                    print_line = print_line + " mp_eventer_yahoo"
                print("-----------{} is alive-----------".format(print_line))

                if self_IP is not None:
                    print("local IP {}".format(self_IP))
                else:
                    self_IP = get_host_ip()
                    try:
                        print("local IP {}".format(self_IP))
                    except:
                        pass

                print("Main CPU: {}%".format(cpu_percent))
                print("Main use RAM: {}".format(ram_percent[3]))
                print("Total RAM: {}".format(ram_percent[0]))
            time.sleep(5)

    def multiprxy_plus_multiprocesses_to_scrapy(self, target, IP, *args):
        multip = {}
        job = {}

        for i, m in enumerate(target):
            print("Job: {} is starting..".format(m))
            job.update({"q{}".format(i): mp.Queue()})
            arg = (m, job["q{}".format(i)],)
            multip.update({"l{}".format(i): mp.Process(target=self.free_ip_get, args=arg)})
            multip["l{}".format(i)].start()

    def free_ip_get(self, target=None or str or dict, qe=None):
        key_dict = {
            "kuaidaili": self.kuaidaili,
            "xicidaili": self.xicidaili,
            "ip3366": self.ip3366,
            "spys_one": self.spys_one,
            "proxyranker": self.proxyranker
        }

        def run_the_head(target):
            res = None
            if target in key_dict:
                res = key_dict[target]()
            elif type(target) is dict:
                res = self.requests_call(target)
            return res

        if not isinstance(target, list):
            target = [target]

        result = []
        for i in target:
            res = run_the_head(i)
            if res is None:
                pass
            else:
                result.append(res)
        if qe is not None:
            print("job: {} is done".format(target))
            qe.put(result)
        else:
            return result

    def requests_call(self, target: dict):
        re = requests.Session()
        params = {}

        if "headers" in target:
            params.update({"headers": target['headers']})

        res = re.get(url=target['url'], headers=params)

        return res.text

    def kuaidaili(self):
        def find_limited_pages():
            headers = self.defults_headers
            headers.update({"User-Agent": self.user_agent_list(),
                            "Host": "www.kuaidaili.com",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
            res = requests.Session().get(url="{}1".format(self.defults_free_ip_web['kuaidaili']), timeout=30,
                                         headers=headers)
            nbs = bs(res.text, "lxml")
            limited_pages = nbs.find_all(id='listnav')
            max_pages = 1
            for i in str(limited_pages).split('id="p'):
                if "</a></li>" in i:
                    data = i.split("</a></li>")[0].split(">")[1]
                    try:
                        tmp = int(data)
                    except:
                        continue
                    if tmp > max_pages:
                        max_pages = tmp
            res = max_pages + 1
            return res

        def requests_(target, headers, queue: mp.Queue):
            headers_ = headers
            headers_.update({"User-Agent": self.user_agent_list(),
                             "Host": "www.kuaidaili.com",
                             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
            response = requests.get(url=target, headers=headers_)
            if response.ok:
                result_sr = response.text
                nbs = bs(result_sr, "lxml")
                data = nbs.find_all(class_='table table-b table-bordered table-striped')
                kuaidaili_IP = []
                for it in str(data).split('<tr>'):
                    if "HTTPS" in str(it) and "IP" in str(it):
                        IP = str(it).split('td data-title="IP">')[1].split('</td>')[0]
                        PORT = str(it).split('td data-title="PORT">')[1].split('</td>')[0]
                        kuaidaili_IP.append("{}: {}".format(IP, PORT))
                queue.put(kuaidaili_IP)
            else:
                return None

        def start_loop_requests(limited):
            headers = self.defults_headers
            kuaidaili_IP = []
            result_td = {}
            local_queue = mp.Queue()
            for i in range(1, limited, 1):
                arg = ("{}{}".format(self.defults_free_ip_web['kuaidaili'], i), headers, local_queue,)
                result_td.update({"td_{}".format(i): td.Thread(target=requests_, args=arg)})
                result_td["td_{}".format(i)].start()

            for i, m in enumerate(result_td.keys()):
                result_td[m].join()
                time.sleep(0.05)

            while True:
                if local_queue.qsize() > 0:
                    local_data = local_queue.get()
                    if local_data is not None:
                        if isinstance(local_data, list):
                            for it in local_data:
                                kuaidaili_IP.append(it)
                        else:
                            kuaidaili_IP.append(local_data)
                    continue
                else:
                    break

            return kuaidaili_IP

        limited = find_limited_pages()
        result = start_loop_requests(limited)

        return result

    def xicidaili(self):
        def find_limited_pages(self):
            headers = self.defults_headers
            headers.update({"User-Agent": self.user_agent_list(),
                            "Host": "www.xicidaili.com",
                            "Accept-Language": "zh-TW,zh;q=0.8,en-US;q=0.5,en;q=0.3",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
            url = "{}1".format(self.defults_free_ip_web['xicidaili'])
            res = requests.Session().get(url=url, headers=headers)
            nbs = bs(res.text, "lxml")
            limited_pages = nbs.find_all(class_='pagination')
            max_pages = 1
            if '<a href="' not in str(limited_pages):
                return None

            for i in str(limited_pages).split('<a href="'):
                if '</a>' in i:
                    tmp = i.split('</a>')[0].split(">")[1]
                else:
                    continue
                if int(tmp) > max_pages:
                    max_pages = int(tmp)
            return max_pages

        def start_requests(self, limited, time_delay=3):
            headers = self.defults_headers
            headers.update({"User-Agent": self.user_agent_list()})
            for i in range(1, limited, 1):
                result_sr = requests.Session().get(url="{}{}".format(self.defults_free_ip_web['xicidaili'], i),
                                                   headers=headers)
                nbs = bs(result_sr.text, "lxml")
                data = nbs.find_all(class_='clearfix proxies')
                for it in str(data).split('<tr'):
                    if '<td>' in it and 'HTTPS' in it:
                        IP = it.split('<td>')[1].split('<')[0]
                        PORT = it.split('<td>')[2].split('<')[0]
                        self.xicidaili_IP.append("{}: {}".format(IP, PORT))
                time.sleep(time_delay)

        limited = find_limited_pages(self)
        result = start_requests(self, limited)

        return result

    def ip3366(self):
        def find_limited_pages():
            headers = self.defults_headers
            headers.update({"User-Agent": self.user_agent_list(),
                            "Host": "www.ip3366.net",
                            "Accept": "text/css,*/*;q=0.1"})
            res = requests.Session().get(url="{}1".format(self.defults_free_ip_web['ip3366']), headers=headers)
            nbs = bs(res.text, "lxml")
            limited_pages = nbs.find_all(id='listnav')
            max_pages = 1
            for i in str(limited_pages).split('href="'):
                if 'page=' in i and '</a>' in i:
                    tmp = i.split('page=')[1].split('</a>')[0].split('>')[1]
                    try:
                        tmp = int(tmp)
                    except:
                        tmp = 0
                        continue
                else:
                    continue
                if int(tmp) > max_pages:
                    max_pages = int(tmp)

            return max_pages

        def requests_(target, headers, queue: mp.Queue):
            ip3366_IP = []
            headers_ = headers
            headers_.update({"User-Agent": self.user_agent_list(),
                             "Host": "www.ip3366.net",
                             "Accept": "text/css,*/*;q=0.1"})
            response = requests.get(url=target, headers=headers_)
            if response.ok:
                result_sr = response.text
                nbs = bs(result_sr, "lxml")
                data = nbs.find_all(class_='table table-bordered table-striped')
                data = bs(str(data)).find_all("tbody")
                data = bs(str(data)).find_all("tr")
                for i in str(data).split("<tr>"):
                    if "<td>" in i and 'HTTPS' in i:
                        IP = i.split("<td>")[1].split('</td>')[0]
                        PORT = i.split("<td>")[2].split('</td>')[0]
                        ip3366_IP.append("{}: {}".format(IP, PORT))
                queue.put(ip3366_IP)
            else:
                return None

        def start_loop_requests(limited):
            headers = self.defults_headers
            ip3366_IP = []
            result_td = {}
            local_queue = mp.Queue()
            for i in range(1, limited + 1, 1):
                arg = ("{}{}".format(self.defults_free_ip_web['ip3366'], i), headers, local_queue,)
                result_td.update({"td_{}".format(i): td.Thread(target=requests_, args=arg)})
                result_td["td_{}".format(i)].start()

            for i, m in enumerate(result_td.keys()):
                result_td[m].join()
                time.sleep(0.05)

            while True:
                t = 0
                if local_queue.qsize() > 0:
                    local_data = local_queue.get()
                    if local_data is not None:
                        if isinstance(local_data, list):
                            for it in local_data:
                                ip3366_IP.append(it)
                        else:
                            ip3366_IP.append(local_data)
                    continue

                for i, m in enumerate(result_td.keys()):
                    if result_td[m].isAlive():
                        t = 1
                        break
                    else:
                        continue

                if t == 0:
                    break

            return ip3366_IP

        limited = find_limited_pages()
        result = start_loop_requests(limited)

        return result

    def spys_one(self):
        def find_limited_pages():
            headers = self.defults_headers
            headers.update({"User-Agent": self.user_agent_list()})
            data = {'xpp': "4",
                    "xf4": "2"}
            res = requests.Session().post(url="{}".format(self.defults_free_ip_web['spys_one']), headers=headers,
                                          data=data)
            nbs = bs(res.text, "lxml")
            limited_pages = nbs.find_all(class_='spy1xx')
            limited_pages.extend(nbs.find_all(class_='spy1x'))

            res = ["{}: 8080".format(str(i).split('class="spy14">')[1].split('<script')[0]) if 'class="spy14">' in str(
                i) and '<script' in str(i) else None for i in limited_pages]
            res = [res.remove(None) if i is None else i for i in res.copy()]

            return res

        result = find_limited_pages()

        return result

    def proxyranker(self):
        result = qe.Queue()
        port = qe.Queue()

        headers = {
            "Connection": "keep-alive",
            "User-Agent": "Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Safari/535.19",
        }
        headers.update({"User-Agent": "Mozilla/5.0 (Android; Mobile; rv:14.0) Gecko/14.0 Firefox/14.0"})

        res = requests.Session().post(url=self.defults_free_ip_web['proxyranker'], headers=headers)
        nbs = bs(res.text, "lxml")
        res = str(nbs.find_all('tr')).split('<th>Port</th>')[-1].split('<td>')

        [result.put(res[i].split('</td>')[0]) if '</td>' in res[i] and "Proxy port" not in res[i] else port.put(
            res[i].split('</td>')[0].split('"Proxy port">')[1].split('</span>')[0]) for i in range(1, len(res))]
        result = ["{}: {}".format(result.get(), port.get()) for i in range(result.qsize())]

        return result

    def test_ip(self, ip: str or list):
        if not isinstance(ip, list):
            ip = [ip]
        result = {}
        result_td = {}
        save_q = mp.Queue()
        for i, m in enumerate(ip):
            arg = (m, save_q)
            result_td.update({"td_{}".format(i): td.Thread(target=self.multiprocessing_for_free_ip, args=arg)})
            result_td["td_{}".format(i)].start()

        for it, imm in enumerate(ip):
            result_td["td_{}".format(it)].join()

        while True:
            if save_q.qsize() > 0:
                data = save_q.get()
                result.update(data)
                time.sleep(0.05)
                continue
            else:
                break

        return result

    def multiprocessing_for_free_ip(self, target, save_q: mp.Queue):
        result = {}
        start_time = time.time()
        re = requests.get(url="https://www.twse.com.tw", proxies={'https//': target})
        end_time = time.time()
        if re.ok:
            s = str(round(float(end_time) - float(start_time), 2)).split('.')[:]
            s1 = s[1]

            if int(s1) < 10:
                s1 = s1 + "0"
            ms = int(str(s[0] + s1))

            if ms > 350:
                return None
            else:
                result.update({target: ms})
                print("Proxy: {} State_code {} Time: {}".format(target, re.status_code, ms))
        save_q.put(result)


if __name__ == "__main__":
    timer: int = 0
    now_time = int(time.time())
    url: str = f'https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_1101.tw&json=1&delay=0&_={now_time}'
    while True:
        start_time = float(time.time())
        try:
            res = requests.session().get(url=url)
            if res.ok is not True: print(res.text)
            print(f'total cost: {float(time.time()) - start_time} ms')
        finally:
            print(timer)
            timer += 1

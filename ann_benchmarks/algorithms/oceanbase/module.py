from time import sleep
import pymysql
import os
import subprocess

from ..base.module import BaseANN

GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE", "/tmp")
OBDSHPATH = os.path.join(GITHUB_WORKSPACE, "oceanbase/tools/deploy/obd.sh")

def metric_mapping(_metric: str):
    _metric_type = {"angular": "vector_cosine_ops",
                    "euclidean": "vector_l2_ops"}.get(_metric, None)
    _metric_func = {"angular": "cosine_distance",
                    "euclidean": "l2_distance"}.get(_metric, None)
    if _metric_type is None or _metric_func is None:
        raise Exception(f"[OceanBase] Not support metric type: {_metric}!!!")
    return _metric_type, _metric_func


class OceanBase(BaseANN):
    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim
        self._metric_type, self._metric_func = metric_mapping(self._metric)
        self.probes = 0
        self.db_config = {}
        self.start_oceanbase()
        max_trys = 10
        for try_num in range(max_trys):
            try:
                self.conn = pymysql.connect(**self.db_config)
            except Exception as e:
                if try_num == max_trys - 1:
                    raise Exception(
                        f"[OceanBase] Connect to oceanbase failed: {e}!!!")
                print(f"[OceanBase] Try to connect to oceanbase again...")
                sleep(2 * try_num)

    def start_oceanbase(self):
        try:
            self.stop_oceanbase()
            result = subprocess.run([OBDSHPATH, 'start'], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if line.startswith("obclient"):
                    for item in line.split():
                        if item.startswith("-h"):
                            self.db_config['host'] = item[2:]
                        elif item.startswith("-P"):
                            self.db_config['port'] = int(item[2:])
                        elif item.startswith("-u"):
                            self.db_config['user'] = item[2:]
                        elif item.startswith("-p"):
                            self.db_config['password'] = item[2:]
                        elif item.startswith("-D"):
                            self.db_config['database'] = item[2:]
                    print(f"[OceanBase] Connect to oceanbase: {self.db_config}")
                    break
            print("[OceanBase] obd start successfully!!!")
        except Exception as e:
            print(f"[OceanBase] obd start failed: {e}!!!")

    def stop_oceanbase(self):
        try:
            os.system(OBDSHPATH + " stop")
            print("[OceanBase] obd stop successfully!!!")
        except Exception as e:
            print(f"[OceanBase] obd stop failed: {e}!!!")

    def create_table(self):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("alter system set syslog_level='ERROR';")
                cursor.execute("alter system set diag_syslog_per_error_limit=1;")
                cursor.execute("alter system set syslog_io_bandwidth_limit='1MB';")
                cursor.execute("alter system set enable_syslog_recycle=True;")
                cursor.execute("drop database if exists vector_bench;")
                cursor.execute("create database vector_bench;")
                cursor.execute("use vector_bench;")
                cursor.execute(
                    "create table items (id int primary key, embedding vector(%d));" % self._dim)
                self.conn.commit()
                print(f"[OceanBase] Table created successfully!!!")
        except Exception as e:
            self.conn.rollback()
            print(f"[OceanBase] Table creation failed: {e}!!!")

    def insert(self, X):
        print(f"[OceanBase] Insert {len(X)} data into the table...")
        try:
            with self.conn.cursor() as cursor:
                cursor.executemany("insert into items (id, embedding) values (%s, %s);", [
                                   (i, str(X[i].tolist())) for i in range(0, len(X))])
                self.conn.commit()
                print(f"[OceanBase] Data has been inserted!!!")
        except Exception as e:
            self.conn.rollback()
            print(f"[OceanBase] Data insertion failed: {e}!!!")

    def get_index_param(self):
        raise NotImplementedError()

    def create_index(self):
        print(f"[OceanBase] Create index for table...")
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    f"create vector index bench_vector_items on items (embedding {self._metric_type}) with({self.get_index_param()});")
                self.conn.commit()
                print(f"[OceanBase] Create index for table successfully!!!")
        except Exception as e:
            self.conn.rollback()
            print(f"[OceanBase] Index creation failed: {e}!!!")

    def fit(self, X):
        self.create_table()
        self.insert(X)
        self.create_index()

    def get_hint(self):
        if self.probes < 1:
            return ""
        return "/*+probes(%d)*/" % self.probes

    def query(self, v, n):
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"select {self.get_hint()} * from items order by {self._metric_func}(embedding, '{v.tolist()}') approx limit {n};")
            result = cursor.fetchall()
            return [row[0] for row in result]

    def done(self):
        self.conn.close()
        self.stop_oceanbase()


class OceanBaseFLAT(OceanBase):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self.name = f"OceanBaseFLAT metric:{self._metric}"

    def create_index(self):
        # FLAT index does not need to create index
        pass


class OceanBaseIVFFLAT(OceanBase):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_lists = index_param.get("lists", 128)

    def get_index_param(self):
        return "type=ivfflat, lists=%d" % self._index_lists

    def set_query_arguments(self, probes):
        self.probes = probes
        self.name = f"OceanBaseIVFFLAT metric:{self._metric}, index_lists:{self._index_lists}, search_probes:{probes}"


class OceanBaseIVFPQ(OceanBase):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_lists = index_param.get("lists", 128)
        self._index_seg = index_param.get("seg", 1)

    def get_index_param(self):
        assert self._dim % self._index_seg == 0, "dimension must be able to be divided by m"
        return "type=ivfpq, lists=%d, seg=%d" % (self._index_lists, self._index_seg)

    def set_query_arguments(self, probes):
        self.probes = probes
        self.name = f"OceanBaseIVFPQ metric:{self._metric}, index_lists:{self._index_lists}, search_probes:{probes}"


class OceanBaseHNSW(OceanBase):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_m = index_param.get("m", 16)
        self._index_ef = index_param.get("ef_construction", 200)
        self.name = f"OceanBaseHNSW metric:{self._metric}, index_M:{self._index_m}, index_ef:{self._index_ef}"

    def get_index_param(self):
        return "type=hnsw, m=%d, ef_construction=%d" % (self._index_m, self._index_ef)

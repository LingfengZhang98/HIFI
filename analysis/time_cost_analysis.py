"""
We evaluate the time cost of each method, and test the effectiveness of GPU acceleration.
This part corresponds to Fig. 10.
"""


import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
import warnings
warnings.filterwarnings('ignore')
import multiprocessing
import time

from bias_mitigation_methods.default import train_original_model
from bias_mitigation_methods.rew import train_with_rew
from bias_mitigation_methods.dir import train_with_dir
from bias_mitigation_methods.fairsmote import train_with_fairsmote
from bias_mitigation_methods.meta import train_with_meta
from bias_mitigation_methods.adv import train_with_adv
from bias_mitigation_methods.pr import train_with_pr
from bias_mitigation_methods.eop import train_with_eop
from bias_mitigation_methods.ceo import train_with_ceo
from bias_mitigation_methods.roc import train_with_roc
from bias_mitigation_methods.maat import train_with_maat
from bias_mitigation_methods.fairmask import train_with_fairmask
from bias_mitigation_methods.hifi import train_with_hifi

from tools.utils import makedirs


if __name__ == '__main__':
    save_root = osp.join(osp.dirname(__file__), "../results")
    makedirs(save_root)
    datasets = ["census", "ufrgs", "compas", "diabetes", "default"]
    seed_range = list(range(0, 10))
    classifiers = ["lr", "dl"]
    n_process = len(seed_range)  # number of parallel processes

    methods_dict = {
        "default": train_original_model,
        "rew": train_with_rew,
        "dir": train_with_dir,
        "fairsmote": train_with_fairsmote,
        "meta": train_with_meta,
        "adv": train_with_adv,
        "pr": train_with_pr,
        "eop": train_with_eop,
        "ceo": train_with_ceo,
        "roc": train_with_roc,
        "maat": train_with_maat,
        "fairmask": train_with_fairmask,
        "hifi": train_with_hifi
    }
    in_methods_list = ["meta", "adv", "pr"]

    fout = open(osp.join(save_root, "time_cost.txt"), "w")

    fout.write("The time cost (sec) of each method on each dataset:\n\n")
    for clf in classifiers:
        fout.write(clf+":\n")
        for dataset in datasets:
            fout.write("\t" + dataset)
        fout.write("\n")

        methods_list = list(methods_dict.keys())
        if clf == "dl":
            methods_list = list(set(methods_list) - set(in_methods_list))

        for method in methods_list:
            if method == "hifi":
                fout.write(method+"_cpu")
                for dataset in datasets:
                    start_time = time.time()
                    with multiprocessing.Pool(processes=n_process) as pool:
                        for seed in seed_range:
                            pool.apply_async(methods_dict[method], args=(dataset, clf, 0.75, [seed], "cpu"))
                        pool.close()
                        pool.join()
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fout.write("\t%.3f" % elapsed_time)
                fout.write("\n")
                fout.write(method + "_gpu")
                for dataset in datasets:
                    start_time = time.time()
                    with multiprocessing.Pool(processes=n_process) as pool:
                        for seed in seed_range:
                            pool.apply_async(methods_dict[method], args=(dataset, clf, 0.75, [seed], "cuda"))
                        pool.close()
                        pool.join()
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fout.write("\t%.3f" % elapsed_time)
            elif method in in_methods_list:
                fout.write(method)
                for dataset in datasets:
                    start_time = time.time()
                    with multiprocessing.Pool(processes=n_process) as pool:
                        for seed in seed_range:
                            pool.apply_async(methods_dict[method], args=(dataset, [seed]))
                        pool.close()
                        pool.join()
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fout.write("\t%.3f" % elapsed_time)
            else:
                fout.write(method)
                for dataset in datasets:
                    if method == "fairsmote" and dataset == "default":
                        fout.write("\ttimeout")
                    else:
                        start_time = time.time()
                        with multiprocessing.Pool(processes=n_process) as pool:
                            for seed in seed_range:
                                pool.apply_async(methods_dict[method], args=(dataset, clf, [seed]))
                            pool.close()
                            pool.join()
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        fout.write("\t%.3f" % elapsed_time)

            fout.write("\n")

        fout.write("\n")

    fout.close()
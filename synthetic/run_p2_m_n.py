import os
import json
import time
import pickle
import copy

import numpy as np

from process_runner import *

parser = argparse.ArgumentParser()
parser.add_argument('--force', default=False,
                    action='store_true', help='force')
parser.add_argument("--max-procs",type=int,default=-1) # -1 for debugging
parser.add_argument("--arr-size",type=int,default=-1)
parser.add_argument("--arr-index",type=int,default=-1)
args = parser.parse_args()

def main():
    max_procs = 4

    cfg = {
        "m" : [50, 70, 100, 150, 200, 300, 400],
        "p" : [2],
        "n" : [50, 100, 200],
        "d" : [1000],
        "noise_scale" : [0.1],
        "r" : [0.1, 0.25],
        "data_seed":range(40),
        "train_seed":range(10),
        'lr' : [1.0, 0.1, 0.01],
    }


    task = MyTask
    runner = MyProcessRunner(
        task,
        cfg,
        max_procs,
        )
    runner.setup()

    runner.run(force=args.force)

    runner.summarize(force=args.force)
    # runner.summarize(force=True)

class MyProcessRunner(ProcessRunner):
    def summarize(self, force = False):
        THRE0 = 0.6

        results_fname = 'outputs/results.pkl'
        if os.path.exists(results_fname) and not force:
            print('loading results from {}'.format(results_fname))
            with open(results_fname, 'rb') as f:
                results = pickle.load(f)
        else:
            print('start reading results...')
            results = {}

            t0 = time.time()

            eof_error_fnames = []

            for t_i, task in enumerate(self.tasks):
                cfg = task.cfg
                del cfg['project_dir']
                del cfg['dataset_dir']
                result_fname1 = task.procs[0].result_fname
                # print(cfg, result_fname)
                with open(result_fname1, 'rb') as f:
                    try:
                        res = pickle.load(f)
                        last_loss = res[-1]['min_dist']
                        key = tuple(cfg.values())
                        # print(key)

                        results[key] = last_loss

                    except EOFError as e:
                        eof_error_fnames.append(result_fname1)


                if t_i % 100 ==0:
                    print(f'reading {t_i}/{len(self.tasks)} done \r' , end = '')

            print('')

            print('removing eof_error files: ..')

            for fname in eof_error_fnames:
                print("remove:", fname)
                os.remove(fname)

            assert len(eof_error_fnames) == 0


            with open(results_fname, 'wb') as f:
                pickle.dump(results, f)

            t1 = time.time()
            print(f'reading and saving results done in {t1-t0:.3f}sec')

        ###### processing ######

        cfg2 = copy.deepcopy(self.cfg)
        del cfg2['data_seed']
        del cfg2['train_seed']
        del cfg2['lr']


        cfgs2 = list(product_dict(**cfg2))

        plot_data = {}

        # print(f"m {cfg2['m'][0]}, r {cfg2['r'][0]}")
        print(f"n {cfg2['n'][0]}")

        for cfg in cfgs2:
            key1 = cfg.keys()
            key1v = cfg.values()

            THRE = THRE0 * cfg["noise_scale"]

            success_rate = 0
            min_values = []
            for d_i in self.cfg['data_seed']:
                is_success = False
                min_value = -1
                for t_i in self.cfg['train_seed']:
                    for lr in self.cfg['lr']:
                        key2 = tuple(list(key1v) + [d_i, t_i, lr])
                        last_value =  results[key2]

                        if np.isnan(last_value):
                            continue

                        if min_value == -1:
                            min_value = last_value
                        elif min_value > last_value:
                            min_value = last_value
                        else:
                            pass

                min_values.append(min_value)

                is_success = (min_value < THRE)
                if is_success:
                    success_rate += 1

            success_rate /= len(self.cfg['data_seed'])

            print(cfg,
                f' min {np.min(min_values):.5f} avg {np.mean(min_values):.5f} max {np.max(min_values):.5f} TH {THRE} sr {success_rate:.3f} ds {len(self.cfg["data_seed"])}')

            # print(cfg,
            #     f' min {np.min(min_values):.5f} avg {np.mean(min_values):.5f} max {np.max(min_values):.5f} ds {len(self.cfg["data_seed"])}')


            plot_data[ (cfg['m'], cfg['n']) ] = success_rate

        # import ipdb; ipdb.set_trace()

        ##### plotting part #####
        data1 = {}
        data1['plot_data'] = plot_data
        data1['cfg'] = self.cfg
        print(plot_data)
        with open("plot_data.pkl", 'wb') as f:
            pickle.dump(data1, f)

        # import matplotlib; matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend; instead, writes files
        # from matplotlib import pyplot as plt

        # noise_scales = self.cfg['noise_scale']
        # rs = self.cfg['r']

        # print("noise_scales")
        # print(noise_scales)
        # print("rs")
        # print(rs)

        # for noise_scale in noise_scales:
        #     success_rates = [ plot_data[(noise_scale,r)] for r in rs ]
        #     print(f"success_rates for noise noise_scale {noise_scale}")
        #     print(success_rates)
        #     plt.plot(rs, success_rates, label = f"noise_scale={noise_scale:.3f}")

        # # plt.title("r vs success rate")
        # plt.xlabel("r")
        # plt.ylabel("success rate")

        # plt.legend(loc="upper left")

        # plt.xscale("log")
        # # plt.yscale("log")

        # plt.savefig("plot.png", dpi=1000)
        # plt.savefig("plot.eps", dpi=1000)



class MyTask(PRTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        env0 = {"OMP_NUM_THREADS":str(1)}

        project_dir = os.path.join('outputs',dict_string(cfg))

        cfg["project_dir"] = project_dir
        cfg["dataset_dir"] = project_dir


        proc1 = PRProcess(
            command = [
                "python", "-u", "gen_data_and_train_cluster.py",
                "--config-override", json.dumps(cfg)
                ],
            output_dir = project_dir,
            result_fname = 'results.pickle',
            cleanup_fnames =[],
            env = env0,
            stdout_prefix = "gen_data_and_train_cluster"
            )

        self.procs = [proc1]



if __name__ == '__main__':

    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---run.py Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))

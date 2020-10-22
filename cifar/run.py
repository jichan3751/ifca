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
parser.add_argument("--config-index",type=int,default=0)
args = parser.parse_args()

def main():
    max_procs = 1
    num_seeds = 5

    cuda_resources = [0]

    cfg = {
        "data_seed":[0,1,2,3,4,5,6,7],
    }

    # cfg["data_seed"] = [0,1,2,4] # maybe to 4,5 later

    task = MyTask
    # taskset_resources = list(range(args.config_index * 40, (args.config_index +1) * 40))

    runner = MyProcessRunner(
        task,
        cfg,
        max_procs,
        cuda_resources =cuda_resources,
        cuda_ct_per_proc =1
        )
    runner.setup()

    # runner.run()
    runner.run(force=args.force)
    # runner.run(force=args.force, verbose=False)

    runner.summarize(force=args.force)
    # runner.summarize(force=True)
    # runner.cleanup()

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
            results1 = {}
            results2 = {}
            results3 = {}

            t0 = time.time()

            eof_error_fnames = []

            for t_i, task in enumerate(self.tasks):
                cfg = task.cfg
                del cfg['project_dir']
                result_fname1 = task.procs[0].result_fname
                result_fname2 = task.procs[1].result_fname
                result_fname3 = task.procs[2].result_fname

                # print(cfg, result_fname)
                with open(result_fname1, 'rb') as f:
                    try:
                        res = pickle.load(f)
                        last_acc = res[-1]['test']['acc']
                        key = tuple(cfg.values())
                        # print(key)

                        results1[key] = last_acc

                    except EOFError as e:
                        eof_error_fnames.append(result_fname1)

                with open(result_fname2, 'rb') as f:
                    try:
                        res = pickle.load(f)
                        last_acc = res[-1]['test']['acc']
                        key = tuple(cfg.values())
                        # print(key)

                        results2[key] = last_acc

                    except EOFError as e:
                        eof_error_fnames.append(result_fname2)

                with open(result_fname3, 'rb') as f:
                    try:
                        res = pickle.load(f)
                        last_acc = res[-10]['test']['acc']
                        key = tuple(cfg.values())
                        # print(key)

                        results3[key] = last_acc

                    except EOFError as e:
                        eof_error_fnames.append(result_fname3)


                if t_i % 100 == 0:
                    print(f'reading {t_i}/{len(self.tasks)} done \r', end = '')


            print('')

            print('removing eof_error files: ..')

            for fname in eof_error_fnames:
                print("remove:", fname)
                os.remove(fname)

            assert len(eof_error_fnames) == 0

            results = [results1,results2,results3]

            with open(results_fname, 'wb') as f:
                pickle.dump(results, f)

            t1 = time.time()
            print(f'reading and saving results done in {t1-t0:.3f}sec')

        ###### processing ######
        print(results[0])

        cfg2 = copy.deepcopy(self.cfg)
        del cfg2['data_seed']

        cfgs2 = list(product_dict(**cfg2))

        plot_data = {}


        for cfg in cfgs2:
            key1 = cfg.keys()
            key1v = cfg.values()

            last_acc0s = []
            last_acc1s = []
            last_acc2s = []

            # import ipdb; ipdb.set_trace()

            for d_i in self.cfg['data_seed']:
                is_success = False
                min_value = -1

                key2 = tuple(list(key1v) + [d_i])
                last_acc0 = results[0][key2]
                last_acc0s.append(last_acc0)

                last_acc1 = results[1][key2]
                last_acc1s.append(last_acc1)

                last_acc2 = results[2][key2]
                last_acc2s.append(last_acc2)

            # print(cfg,
                # f' IFCA {np.mean(last_acc0s):.5f} Global {np.mean(last_acc1s):.5f} Local {np.mean(last_acc2s):.5f} ds {len(self.cfg["data_seed"])}')

            print(cfg,
                f' IFCA {np.mean(last_acc0s):.5f} +- {np.std(last_acc0s):.5f} Global {np.mean(last_acc1s):.5f} +- {np.std(last_acc1s):.5f} Local {np.mean(last_acc2s):.5f} +- {np.std(last_acc2s):.5f} ds {len(self.cfg["data_seed"])}')

            # plot_data[ (cfg['n']) ] = acc


class MyTask(PRTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        env0 = {"OMP_NUM_THREADS":str(4)}

        project_dir = os.path.join('outputs',dict_string(cfg))

        cfg["project_dir"] = project_dir

        self.procs = []

        proc = PRProcess(
            command = [
                "python", "-u", "train_cluster_cifar_tf.py",
                "--config-override", json.dumps(cfg)
                ],
            output_dir = project_dir,
            result_fname = 'results.pickle',
            cleanup_fnames =[],
            env = env0,
            stdout_prefix = "train_cluster_cifar_tf"
            )

        self.procs.append(proc)

        proc = PRProcess(
            command = [
                "python", "-u", "train_cluster_cifar_tf_single.py",
                "--config-override", json.dumps(cfg)
                ],
            output_dir = project_dir,
            result_fname = 'results_single.pickle',
            cleanup_fnames =[],
            env = env0,
            stdout_prefix = "train_cluster_cifar_tf_single"
            )
        self.procs.append(proc)

        proc = PRProcess(
            command = [
                "python", "-u", "train_cluster_cifar_tf_local.py",
                "--config-override", json.dumps(cfg)
                ],
            output_dir = project_dir,
            result_fname = 'results_local.pickle',
            cleanup_fnames =[],
            env = env0,
            stdout_prefix = "train_cluster_cifar_tf_local"
            )
        self.procs.append(proc)


if __name__ == '__main__':

    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---run.py Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))

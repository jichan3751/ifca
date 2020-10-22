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
    max_procs = 5
    num_seeds = 5

    if args.config_index ==0:
        cfg = {
            "m" : [2400],
            "m_test" : [400],
            "p" : [4],
            "n" : [100],
            "data_seed":range(num_seeds),
        }
    elif args.config_index ==1:
        cfg = {
            "m" : [1200],
            "m_test" : [200],
            "p" : [4],
            "n" : [200],
            "data_seed":range(num_seeds),
        }
    elif args.config_index ==2:
        cfg = {
            "m" : [4800],
            "m_test" : [800],
            "p" : [4],
            "n" : [50],
            "data_seed":range(num_seeds),
        }
    else:
        assert 0

    task = MyTask


    runner = MyProcessRunner(
        task,
        cfg,
        max_procs,
        )
    runner.setup()

    runner.run(force=args.force)

    runner.summarize(force=args.force)
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
            results4 = {}

            t0 = time.time()

            eof_error_fnames = []

            for t_i, task in enumerate(self.tasks):
                cfg = task.cfg
                del cfg['project_dir']
                result_fname1 = task.procs[0].result_fname
                result_fname2 = task.procs[1].result_fname
                result_fname3 = task.procs[2].result_fname
                result_fname4 = task.procs[3].result_fname
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
                        last_acc = res[-1]['test']['acc']
                        key = tuple(cfg.values())
                        # print(key)

                        results3[key] = last_acc

                    except EOFError as e:
                        eof_error_fnames.append(result_fname3)

                with open(result_fname4, 'rb') as f:
                    try:
                        res = pickle.load(f)
                        last_acc = res[-1]['test']['acc']
                        key = tuple(cfg.values())
                        # print(key)

                        results4[key] = last_acc

                    except EOFError as e:
                        eof_error_fnames.append(result_fname4)

                if t_i % 100 == 0:
                    print(f'reading {t_i}/{len(self.tasks)} done \r', end = '')


            print('')

            print('removing eof_error files: ..')

            for fname in eof_error_fnames:
                print("remove:", fname)
                os.remove(fname)

            assert len(eof_error_fnames) == 0

            results = [results1,results2,results3,results4]

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
            last_acc3s = []

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

                last_acc3 = results[3][key2]
                last_acc3s.append(last_acc3)


            # print(cfg,
            #     f' IFCA {np.mean(last_acc0s):.5f} Global {np.mean(last_acc1s):.5f} Local {np.mean(last_acc2s):.5f} Finetune {np.mean(last_acc3s):.5f} ds {len(self.cfg["data_seed"])}')

            print(cfg,
                f' IFCA {np.mean(last_acc0s):.5f} +- {np.std(last_acc0s):.5f} Global {np.mean(last_acc1s):.5f} +- {np.std(last_acc1s):.5f} Local {np.mean(last_acc2s):.5f} +- {np.std(last_acc2s):.5f} ds {len(self.cfg["data_seed"])}')


            # plot_data[ (cfg['n']) ] = acc


class MyTask(PRTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        env0 = {"OMP_NUM_THREADS":str(4)}

        project_dir = os.path.join('outputs',dict_string(cfg))

        cfg["project_dir"] = project_dir

        proc1 = PRProcess(
            command = [
                "python", "-u", "train_cluster_mnist.py",
                "--config-override", json.dumps(cfg)
                ],
            output_dir = project_dir,
            result_fname = 'results.pickle',
            cleanup_fnames =[],
            env = env0,
            stdout_prefix = "train_cluster_mnist"
            )

        proc2 = PRProcess(
            command = [
                "python", "-u", "train_cluster_mnist_single.py",
                "--config-override", json.dumps(cfg)
                ],
            output_dir = project_dir,
            result_fname = 'results_single.pickle',
            cleanup_fnames =[],
            env = env0,
            stdout_prefix = "train_cluster_mnist_single"
            )

        proc3 = PRProcess(
            command = [
                "python", "-u", "train_cluster_mnist_local.py",
                "--config-override", json.dumps(cfg)
                ],
            output_dir = project_dir,
            result_fname = 'results_local.pickle',
            cleanup_fnames =[],
            env = env0,
            stdout_prefix = "train_cluster_mnist_local"
            )

        proc4 = PRProcess(
            command = [
                "python", "-u", "train_cluster_mnist_local.py",
                "--checkpoint", str(1),
                "--config-override", json.dumps(cfg)
                ],
            output_dir = project_dir,
            result_fname = 'results_ckpt_local.pickle',
            cleanup_fnames =[],
            env = env0,
            stdout_prefix = "train_cluster_mnist_ckpt_local"
            )

        self.procs = [proc1, proc2, proc3, proc4]

if __name__ == '__main__':

    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---run.py Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))

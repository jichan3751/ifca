import os
import argparse
import itertools
import subprocess

def main():
    max_procs = 3

    cfg = {
        'm' : [100, 200],
        'n' : [5, 10],
        'lr' : [0.1, 0.01],
    }

    task = SimpleEchoTask
    runner = ProcessRunner(task, cfg, max_procs)
    runner.setup()
    runner.run()


#### Low Level Components ###

class ProcessRunner(object):
    """Runs multiple task in parallel"""
    def __init__(
            self,
            task_class,
            cfg, max_procs = 1,
            arr_size = None,
            arr_index = None,
            taskset_resources = None,
            taskset_ct_per_proc = None,
            cuda_resources = None,
            cuda_ct_per_proc = None,
            ):
        self.task = task_class
        self.cfg = cfg
        self.max_procs = max_procs
        self.arr_size = arr_size
        self.arr_index = arr_index

        if taskset_resources:
            assert isinstance(taskset_resources, list)
            assert len(taskset_resources) % taskset_ct_per_proc == 0
            assert self.max_procs == (len(taskset_resources)//taskset_ct_per_proc)
            self.taskset_resources = chunkify(taskset_resources, len(taskset_resources)//taskset_ct_per_proc)
        else:
            self.taskset_resources = None


        if cuda_resources:
            assert isinstance(cuda_resources, list)
            assert len(cuda_resources)%cuda_ct_per_proc == 0
            assert self.max_procs == (len(cuda_resources)//cuda_ct_per_proc)
            self.cuda_resources = chunkify(cuda_resources, len(cuda_resources)//cuda_ct_per_proc)
        else:
            self.cuda_resources = None


    def setup(self):

        self.cfgs = list(product_dict(**self.cfg))

        if self.arr_size is not None and self.arr_index is not None:
            assert arr_size > arr_index
            self.cfgs = chunk(self.cfgs, self.arr_index ,self.arr_size)

        self.tasks = [ self.task(cfg) for cfg in self.cfgs]

        for task in self.tasks:
            task.setup()

    def run(self, force=False, verbose=True):
        resource_list = []
        next_resource = 0
        task_list = []
        for task_i,task in enumerate(self.tasks):
            if verbose: print(f'Running task {task_i}/{len(self.tasks)} ct {task.current_proc}')

            if self.taskset_resources:
                task.set_taskset_resource(self.taskset_resources[next_resource])

            if self.cuda_resources:
                task.set_cuda_resource(self.cuda_resources[next_resource])

            task.run_next(force)
            task_list.append(task)
            resource_list.append(next_resource)
            if len(task_list) >= self.max_procs:
                launch_next = False
                while not launch_next:
                    for j, task in enumerate(task_list):
                        task.wait()
                        if task.is_done():
                            launch_next = True
                            task_list.pop(j)
                            next_resource = resource_list.pop(j)
                            break
                        else:
                            # print(f'running task {j} ct {task.current_proc}')
                            task.run_next(force)
            else:
                next_resource += 1


        while len(task_list) > 0:
            launch_next = False
            while not launch_next:
                for j, task in enumerate(task_list):
                    task.wait()
                    if task.is_done():
                        launch_next = True
                        task_list.pop(j)
                        next_resource = resource_list.pop(j)
                        break
                    else:
                        # print(f'running task {j} ct {task.current_proc}')
                        task.run_next(force)

    def cleanup(self):
        """ remove big intermediate files """
        for i,task in enumerate(self.tasks):
            task.cleanup()

    def summarize(self):
        """ custom function to show summarization of the results. to be inherited """
        for task in self.tasks:
            cfg = task.cfg
            result_fname = task.procs[1].result_fname
            print(cfg, result_fname)
            # with open(result_fname) as json_file:
            #     results = json.load(json_file)

        raise NotImplementedError

class PRTask(object):
    """
        generates process seqeuence to be run, maintains states when running

    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.procs = None

        self.taskset_resource = None
        self.cuda_resource = None

    def setup(self):
        self.current_proc = 0

    def set_taskset_resource(self, l1):
        assert isinstance(l1, list)
        self.taskset_resource = l1

    def set_cuda_resource(self, l1):
        assert isinstance(l1, list)
        self.cuda_resource = l1


    def run_next(self, force=False):
        proc = self.procs[self.current_proc]
        proc.run(
            force = force,
            taskset = self.taskset_resource,
            cuda = self.cuda_resource,
            )

    def wait(self):
        proc = self.procs[self.current_proc]
        proc.wait()
        self.current_proc += 1

    def is_done(self):
        return len(self.procs) == self.current_proc

    def cleanup(self):
        for proc in self.procs:
            proc.cleanup()






class PRProcess(object):
    """docstring for PRProcess
        This is signature !!

    """
    def __init__(self, command, output_dir, result_fname, cleanup_fnames=None, env = None, stdout_prefix = ""):
        # TODO: some assertions?

        self.command = command
        self.output_dir = output_dir
        self.result_fname =  os.path.join(self.output_dir ,result_fname)
        self.cleanup_fnames = cleanup_fnames
        self.env = env
        self.stdout_prefix = stdout_prefix

        self.is_running = False

    def run(self, force = False, verbose = False, taskset = None, cuda = None):
        if not os.path.exists(self.result_fname) or force:

            os.makedirs(self.output_dir, exist_ok=True)

            my_env = os.environ.copy()
            if self.env is not None:
                my_env.update(self.env)

            if cuda is not None:
                str_indices = [ str(i) for i in cuda]
                my_env["CUDA_VISIBLE_DEVICES"] = ",".join(str_indices)

            stderr_fname = os.path.join(self.output_dir , self.stdout_prefix+".stderr.txt")
            stdout_fname = os.path.join(self.output_dir , self.stdout_prefix+".stdout.txt")

            self.stdout_file = open(stdout_fname, 'w')
            self.stderr_file = open(stderr_fname, 'w')

            if taskset is not None:
                str_indices = [ str(i) for i in taskset]
                command = ['taskset', '-c', ",".join(str_indices)] + self.command
            else:
                command = self.command


            self.process = subprocess.Popen(command, stdout=self.stdout_file, stderr = self.stderr_file, env=my_env)
            self.is_running = True

            print("==>Running:", " ".join(command))
            print(f"   track -> tail -f {stdout_fname}")


        else:
            if verbose:
                print("==>Skipping:", " ".join(self.command))

    def wait(self):
        if self.is_running:
            print("=!>Waiting:", " ".join(self.command))
            self.process.wait()
            self.stdout_file.close()
            self.stderr_file.close()
            self.is_running=False

    def cleanup(self):
        if self.cleanup_fnames is None:
            return
        else:
            for fname in self.cleanup_fnames:
                remove_fname = os.path.join(self.output_dir ,fname)
                if os.path.exists(remove_fname):
                    os.remove(remove_fname)





# define what kind of process that I will run here...
class SimpleEchoTask(PRTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        proc1 = PRProcess(
            command = ["echo", str(cfg['m']),str(cfg['n']),str(cfg['lr'])],
            output_dir = f"outputs/echo_{cfg['m']}_{cfg['n']}_{cfg['lr']}_1",
            result_fname = 'stdout.txt',
            cleanup_fnames =['stdout.txt','stderr.txt']
            )

        proc2 = PRProcess(
            command = ["sleep", "0.2"],
            output_dir = f"outputs/echo_{cfg['m']}_{cfg['n']}_{cfg['lr']}_sleep",
            result_fname = 'stdout.txt',
            cleanup_fnames =['stdout.txt','stderr.txt']
            )

        proc3 = PRProcess(
            command = ["echo", str(cfg['m']),str(cfg['n']),str(cfg['lr']), "22"],
            output_dir = f"outputs/echo_{cfg['m']}_{cfg['n']}_{cfg['lr']}_2",
            result_fname = 'stdout.txt',
            cleanup_fnames =['stdout.txt','stderr.txt']
            )

        self.procs = [proc1, proc2, proc3]
        # self.procs = [proc1, proc3]




### utils ###

def dict_string(my_dict):
    str_list = []
    for key in my_dict:
        if type(my_dict[key]) == float:
            str_list.append(key + f"_{my_dict[key]:.6f}")
        else:
            str_list.append(key + f"_{my_dict[key]}")

    return "_".join(str_list)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

    # use: list(product_dict(**mydict))


def chunk(a, i, n):
    a2 = chunkify(a, n)
    return a2[i]

def chunkify(a, n):
    # splits list into even size list of lists
    # [1,2,3,4] -> [1,2], [3,4]

    k, m = divmod(len(a), n)
    gen = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(gen)


if __name__ == '__main__':
    main()

import numpy as np
import time

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

class Server:

    # def __init__(self, client_model):
    #     self.client_model = client_model
    #     self.model = client_model.get_params()
    #     self.selected_clients = []
    #     self.updates = []

    def __init__(self, client_models):
        self.client_models = client_models
        self.models = [client_model.get_params() for client_model in client_models]

        self.num_groups = len(client_models)

        self.selected_clients = []
        self.updates = []
        self.updates_group_infos = []

    def one_shot_clustering(self, clients, seed):
        NUM_CLASSES = 62
        t0 = time.time()

        self.train_group_index = {}

        num_clients = len(clients)

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(solver='liblinear',multi_class='auto',random_state=seed)

        params = []
        for c_i, client in enumerate(clients):
            # if c_i>3: # d
            #     break

            t00 = time.time()
            X = np.array(client.train_data['x'])
            y = np.array(client.train_data['y'])

            clf.fit(X, y)
            acc = clf.score(X,y)
            param = np.copy(clf.coef_)

            local_labels_active = [0 for _ in range(NUM_CLASSES)]
            for y_el in y:
                local_labels_active[y_el] = 1

            local_labels = []
            for i, e in enumerate(local_labels_active):
                if e:
                    local_labels.append(i)

            new_param = np.zeros((NUM_CLASSES,param.shape[1]))
            for i, label in enumerate(local_labels):
                new_param[label,:] = param[i,:]

            params.append(new_param.flatten())
            t01 = time.time()
            # print(param.shape)
            print(f"  logistic regression {c_i}/{num_clients}, acc {acc:.3f} took {t01-t00:.3f} sec\r", end="")



        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.num_groups, random_state=seed)
        kmeans.fit(params) # fit on train data
        cluster_assign = kmeans.predict(params)

        t1 = time.time()

        print(f"logistic regression + kmeans took {t1-t0:.3f}sec")

        # import ipdb; ipdb.set_trace()


        for c_i, client in enumerate(clients):
            self.train_group_index[client.id] = cluster_assign[c_i]

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, force_group_dict = None):
        """Trains self.model on given clients.

        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}

        client_test_metrics = self.test_model_group_info(clients, 'train')

        for c in clients:
            group_index = client_test_metrics[c.id]['group_index'] ##IFCA

            if force_group_dict is not None:
                # print(c.id,"c",group_index,'f',force_group_dict[c.id])
                group_index = force_group_dict[c.id]
            else:
                # print(c.id,"c",group_index)
                pass

            c.model.set_params(self.models[group_index])
            comp, num_samples, update = c.train(num_epochs, batch_size, minibatch)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((num_samples, update))
            self.updates_group_infos.append(group_index)

        return sys_metrics

    def update_model(self):
        # total_weight = 0.

        # base = [0] * len(self.updates[0][1])

        # for (client_samples, client_model) in self.updates:
        #     total_weight += client_samples
        #     for i, v in enumerate(client_model):
        #         base[i] += (client_samples * v.astype(np.float64))

        # averaged_soln = [v / total_weight for v in base]

        # self.model = averaged_soln
        # self.updates = []


        ### ifca
        updates_by_group = [[] for _ in range(self.num_groups)]

        for i, g_i in enumerate(self.updates_group_infos):
            updates_by_group[g_i].append(self.updates[i])

        for g_i, updates in enumerate(updates_by_group):
            if len(updates) == 0:
                continue

            # ## default
            total_weight = 0.
            base = [0] * len(updates[0][1])

            for (client_samples, client_model) in updates:
                total_weight += client_samples
                for i, v in enumerate(client_model):
                    base[i] += (client_samples * v.astype(np.float64))

            averaged_soln = [v / total_weight for v in base]



            ## faster?

            # num_weights = len(updates[0][1])

            # samples_arr = []
            # ws = [[] for _ in range(num_weights)]
            # for (client_samples, client_model) in updates:
            #     samples_arr.append(client_samples)
            #     for i, v in enumerate(client_model):
            #         ws[i].append(v)

            # averaged_soln2 = [ np.average(ws[i], axis = 0, weights = samples_arr ) for i in range(num_weights)]
            # averaged_soln = averaged_soln2
            # import ipdb; ipdb.set_trace() # confirmed same

            self.models[g_i] = averaged_soln


        ### ifca end
        self.updates = []
        self.updates_group_infos = []

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """


        if clients_to_test is None:
            clients_to_test = self.selected_clients

        metrics = self.test_model_group_info(clients_to_test, set_to_use)


        return metrics

    def test_model_group_info(self, clients_to_test, set_to_use='test'):
        metrics = {}
        for client in clients_to_test:
            tmp_c_metrics = []

            g_i = self.train_group_index[client.id] # use train's cluster info

            client.model.set_params(self.models[g_i])
            c_metrics = client.test(set_to_use)

            c_metrics['group_index'] = g_i

            metrics[client.id] = c_metrics


        return metrics


    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_models[0].set_params(self.model[0])
        model_sess =  self.client_models[0].sess
        return self.client_models[0].saver.save(model_sess, path)

    def close_model(self):
        # self.client_model.close()
        for client_model in self.client_models:
            client_model.close()
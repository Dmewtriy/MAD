class Cluster:
	def __init__(self):
		self.S = 0 # total number of objects
		self.W = 0 # width of histogram
		self.N = 0 # number of transactions
		self.Occ = {} # occurrence of objects

	def add_transaction(self, transaction):
		for item in transaction:
			self.Occ[item] = self.Occ.get(item, 0) + 1
		self.S += len(transaction)
		self.W = len(self.Occ)
		self.N += 1

	def remove_transaction(self, transaction):
		for item in transaction:
			self.Occ[item] = self.Occ.get(item, 0) - 1
			if self.Occ[item] == 0:
				del self.Occ[item]
		self.S -= len(transaction)
		self.W = len(self.Occ)
		self.N -= 1

	def delta_add(self, transaction, r):
		S_new = self.S + len(transaction)
		W_new = self.W
		for item in transaction:
			if self.Occ.get(item, 0) == 0:
				W_new += 1
		if(self.W == 0):
			return S_new * (self.N + 1) / (W_new ** r)
		else:
			return S_new * (self.N + 1) / (W_new ** r) - self.S * self.N / (self.W ** r)


class CLOPE:
	def __init__(self, number_of_transactions, get_next_transaction, save_number_of_cluster, r):
		self.clusters = [] # current list of clusters
		self.r = r # repulsion value
		self.number_of_transactions = number_of_transactions # total number of transaction
		self.get_next_transaction = get_next_transaction	# function, returns transaction id, transaction(list) and number of cluseter
		self.save_number_of_cluster = save_number_of_cluster	# function, gets transaction id and number of cluster

	# inmlementation of the CLOPE algorithm
	def exec(self):
		# Phrase 1 - Initialization
		first = Cluster()
		(id, t, n) = self.get_next_transaction()
		first.add_transaction(t)
		self.clusters.append(first)
		self.save_number_of_cluster(id, 0)
		for _ in range(self.number_of_transactions - 1):
			obj = self.get_next_transaction()
			if obj == None:
				return
			(id, t, n) = obj
			empty_cluster = Cluster()
			max_delta = empty_cluster.delta_add(t, self.r)
			max_delta_index = -1
			for i, c in enumerate(self.clusters):
				delta = c.delta_add(t, self.r)
				if delta > max_delta:
					max_delta = delta
					max_delta_index = i
			if max_delta_index == -1:
				new_cluster = Cluster()
				new_cluster.add_transaction(t)
				self.clusters.append(new_cluster)
				self.save_number_of_cluster(id, len(self.clusters) - 1)
			else:
				self.clusters[max_delta_index].add_transaction(t)
				self.save_number_of_cluster(id, max_delta_index)
		# Phrase 2 - Iteration
		moved = True
		self.n_iter_ = 0
		while moved == True:
			self.n_iter_ += 1
			if self.n_iter_ > 100:
				raise RuntimeError("CLOPE did not converge in 100 passes")
			moved = False
			for _ in range(self.number_of_transactions):
				(id, t, n) = self.get_next_transaction()
				self.clusters[n].remove_transaction(t)
				empty_cluster = Cluster()
				max_delta = empty_cluster.delta_add(t, self.r)
				max_delta_index = -1
				old_delta = self.clusters[n].delta_add(t, self.r)
				if old_delta >= max_delta:
					max_delta = old_delta
					max_delta_index = n
				for i, c in enumerate(self.clusters):
					delta = c.delta_add(t, self.r)
					if delta > max_delta:
						max_delta = delta
						max_delta_index = i
				if max_delta_index == -1:
					new_cluster = Cluster()
					new_cluster.add_transaction(t)
					self.clusters.append(new_cluster)
					self.save_number_of_cluster(id, len(self.clusters) - 1)
					moved = True
				else:
					self.clusters[max_delta_index].add_transaction(t)
					if max_delta_index != n:
						self.save_number_of_cluster(id, max_delta_index)
						moved = True

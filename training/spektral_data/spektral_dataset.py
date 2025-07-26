from spektral.data import Dataset, Graph

class MyDataset(Dataset):
    def __init__(self, graphs_data, labels_data, **kwargs):
        
        self.graphs_data = graphs_data # Lista Spektral Graph objekata
        self.labels_data = labels_data # One-hot enkodirane labele
        
        print(f"Loaded {len(self.graphs_data)} graphs for MyDataset")
        super().__init__(**kwargs)

    def read(self):
        output_graphs = []
        for i, graph_obj in enumerate(self.graphs_data):
            # Ažuriramo y atribut grafa sa one-hot enkodiranom labelom
            graph_obj.y = self.labels_data[i]
            output_graphs.append(graph_obj)
        return output_graphs

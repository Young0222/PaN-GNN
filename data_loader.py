import pandas as pd

general_file = '/home/xjc/222pane-gnn/SiReN-reco-main-incremental/'
general_file_kuairec = '/home/xjc/222pane-gnn/SiReN-reco-main-kuairec/'

class Data_loader():
    def __init__(self,dataset,version):
        self.dataset=dataset; self.version=version
        if dataset=='ML-1M':
            self.sep='::'
            self.names=['userId','movieId','rating','timestemp']
            
            self.path_for_whole = general_file + 'ml-1m/ratings.dat'
            self.path_for_train = general_file + '/ml-1m/train_1m%s.dat'%(version)
            # self.path_for_train = general_file + '/ml-1m/train_1m%s.dat.demo'%(version)
            # self.path_for_test = general_file + '/ml-1m/test_1m%s.dat'%(version)
            # self.path_for_test = general_file + '/ml-1m/test_1m%s.dat.demo'%(version)
            self.path_for_test = general_file + '/ml-1m/test_1m%s.sparse.g1.dat'%(version)
            self.num_u=6040; self.num_v=3952;
            
        
        elif dataset=='amazon':
            self.path_for_whole = general_file + '/amazon-book/amazon-books-enc.csv'
            self.path_for_train = general_file + '/amazon-book/train_amazon%s.dat'%(version)
            self.path_for_test = general_file + '/amazon-book/test_amazon%s.dat'%(version)
            self.num_u=35736; self.num_v=38121;

        elif dataset=='yelp':
            self.path_for_whole = general_file + '/yelp/YELP_encoded.csv'
            self.path_for_train = general_file + '/yelp/train_yelp%s.dat'%(version)
            self.path_for_test = general_file + '/yelp/test_yelp%s.dat'%(version)
            self.num_u=41772; self.num_v=30037;

        elif dataset=='kwai_bias_noremap':
            self.names=['userId','movieId','rating']
            self.path_for_train = general_file_kuairec + '/kwai_bias_noremap/train.dat'
            self.path_for_test = general_file_kuairec + '/kwai_bias_noremap/test.dat'
            self.num_u=7176
            self.num_v=10728
        
        elif dataset=='kwai_unbias_noremap':
            self.names=['userId','movieId','rating']
            self.path_for_train = general_file_kuairec + '/kwai_unbias_noremap/train.dat'
            self.path_for_test = general_file_kuairec + '/kwai_unbias_noremap/test.dat'
            self.num_u=7176
            self.num_v=10728

        else:
            raise NotImplementedError("incorrect dataset, you can choose the dataset in ('ML-100K','ML-1M','amazon','yelp')")
        
        

    def data_load(self):
        if self.dataset=='ML-1M':
            self.whole_=pd.read_csv(self.path_for_whole, names = self.names, sep=self.sep, engine='python').drop('timestemp',axis=1).sample(frac=1,replace=False,random_state=self.version)
            self.train_set = pd.read_csv(self.path_for_train,engine='python',names=self.names).drop('timestemp',axis=1)
            self.test_set = pd.read_csv(self.path_for_test,engine='python',names=self.names).drop('timestemp',axis=1)            
                
        elif self.dataset=='amazon':
            self.whole_=pd.read_csv(self.path_for_whole,index_col=0).sample(frac=1,replace=False);
            self.train_set=pd.read_csv(self.path_for_train,index_col=0)
            self.test_set=pd.read_csv(self.path_for_test,index_col=0)
            

        elif self.dataset=='yelp':
            self.whole_=pd.read_csv(self.path_for_whole,index_col=0).sample(frac=1,replace=False);
            self.train_set=pd.read_csv(self.path_for_train,index_col=0)
            self.test_set=pd.read_csv(self.path_for_test,index_col=0)

        elif self.dataset=='kwai_bias_noremap':
            self.train_set = pd.read_csv(self.path_for_train,engine='python',names=self.names)
            self.test_set = pd.read_csv(self.path_for_test,engine='python',names=self.names)   

        elif self.dataset=='kwai_unbias_noremap':
            self.train_set = pd.read_csv(self.path_for_train,engine='python',names=self.names)
            self.test_set = pd.read_csv(self.path_for_test,engine='python',names=self.names) 
        
        return self.train_set, self.test_set
    
    
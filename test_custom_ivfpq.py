import pickle
import numpy as np
import pytest
from custom_ivfpq import CustomIndexIVFPQ
    
@pytest.fixture()
def index():
    d = 128
    m = 8
    nlist = 10
    nbits = 8
    return CustomIndexIVFPQ(d,m,nlist,nbits)

@pytest.fixture()
def data():
    return np.random.random((300,128))

@pytest.fixture()
def trained_index(index, data):
    index.train(data)
    return index 

class TestIndexApi:
    def test_cannot_add_before_train(self, index, data):
        with pytest.raises(ValueError):
            index.add(data)
            
    def test_cannot_search_before_train(self, index, data):
        with pytest.raises(ValueError):
            index.search(data, 5)
            
    def test_cannot_search_before_add(self, trained_index, data):
        with pytest.raises(ValueError):
            trained_index.search(data, 5)
            
class TestInitializationParameters:
    def test_d_multiple_m(self):
        d = 128
        m = 7
        nlist = 10
        nbits = 8
        with pytest.raises(ValueError):
            CustomIndexIVFPQ(d,m,nlist,nbits)
    def test_nbits_not_allowed(self):
        d = 128
        m = 7
        nlist = 10
        nbits = 4
        with pytest.raises(ValueError):
            CustomIndexIVFPQ(d,m,nlist,nbits)
            
class TestFeatureDimensions():
    def test_caltech(self):
        with open('features/features-caltech101-resnet.pickle','rb') as f:
            feature_list = pickle.load(f)
        assert feature_list.shape[1] == 2048
    def test_caltech(self):
        with open('features/features-voc2012-resnet.pickle','rb') as f:
            feature_list = pickle.load(f)
        assert feature_list.shape[1] == 2048
                
def test_id_increase_on_multiple_add(data, index):
    index.train(data)
    first_id = index.max_id
    index.add(data)
    second_id = index.max_id
    assert second_id > first_id
    
def test_find_smallest_k(index):
    distances = [5,9,7,4,1,0,6,3,2]
    filtered_ids = np.arange(10)
    _, I = index.find_smallest_k(distances, filtered_ids, 3)
    
    assert I == (5,4,8)
    
    
##################################
### Simulate the list of spark ###
##################################
class SList(list):
    def group_by(self, key_function):
        '''
            Similar to the semantics of spark
            Input: key
            Output: An new s-list of grouped data points based on key
        '''
        result = {}
        for item in self:
            key = key_function(item)
            if result.has_key(key):
                result[key].append(item)
            else:
                result[key] = [item]
        return SList([(key, SList(result[key])) for key in result])

    def get_element_count(self, key):
        '''
            Get a count number of distinct values for a given key
            Input: key
            Output: An integer of the number of distinct values. -1 if the key if wrong or data_set.
        '''
        if len(self) == 0:
            return 0
        if not self[0].has_key(key):
            return -1
        tem_value_list  = [item[key] for item in self]
        return len(set(tem_value_list))

    def sort_by(self, sort_func):
        '''
            Sort the dataset by given function
            Input: sorting function
            Output: A new dataset of sorted results
        '''
        return SList(sorted(self, key=sort_func))

    def count(self):
        '''
            Return the total number of elements in dataset
            Output: interger
        '''
        return len(self)

    def filter_by(self, filter_func):
        '''
            Return a filtered dataset based on filter_func
            Semantics are similar to spark
            Input: filter function
            Output: filtered dataset
        '''
        return SList(filter(filter_func, self))

    def map(self, map_func):
        '''
            Semantics are similar to spark
            Input: map function
            Output: dataset after applying map function for each item
        '''
        return SList([map_func(item) for item in self])

    def flatmap(self, map_func):
        '''
            Semantics are similar to spark
            Input: map function
            Output: dataset
        '''
        result = SList()
        for item in self:
            result += map_func(item)
        return result

    def distinct(self):
        '''
            Return a distinct element set
            Note: The order will be changed
        '''
        return SList(list(set(self)))

    def find_by(self, target_func):
        '''
            Return the first item fits target_func
            Return False if nothing fits
        '''
        for item in self:
            if target_func(item):
                return item
        return False
